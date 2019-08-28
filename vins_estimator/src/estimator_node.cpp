#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"


Estimator estimator;

//https://www.jianshu.com/p/c1dfa1d40f53
std::condition_variable con;//条件变量
double current_time = -1;

queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
// 订阅pose graph node发布的回环帧数据，存到relo_buf队列中，供重定位使用
queue<sensor_msgs::PointCloudConstPtr> relo_buf;

int sum_of_wait = 0;//这个变量没有使用

//互斥量
std::mutex m_buf; // 用于处理多个线程使用imu_buf和feature_buf的冲突
std::mutex m_state; // 用于处理多个线程使用当前里程计信息（即tmp_P、tmp_Q、tmp_V）的冲突
std::mutex i_buf;
std::mutex m_estimator; // 用于处理多个线程使用VINS系统对象（即Estimator类的实例estimator）的冲突

double latest_time; // 最近一次里程计信息对应的IMU时间戳

// 当前里程计信息
//IMU项[P,Q,B,Ba,Bg,a,g]
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;

// 当前里程计信息对应的IMU bias
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;

// 当前里程计信息对应的IMU测量值
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;


bool init_feature = 0;
bool init_imu = 1; // true：第一次接收IMU数据
double last_imu_t = 0; // 最近一帧IMU数据的时间戳

//从IMU测量值imu_msg和上一个PVQ递推得到下一个tmp_Q，tmp_P，tmp_V，中值积分
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();

    //init_imu=1表示第一个IMU数据
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;
        return;
    }

    double dt = t - latest_time;
    latest_time = t;

    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

//从估计器中得到滑动窗口当前图像帧的imu更新项[P,Q,V,ba,bg,a,g]
//对imu_buf中剩余的imu_msg进行PVQ递推
// 当处理完measurements中的所有数据后，如果VINS系统正常完成滑动窗口优化，那么需要用优化后的结果更新里程计数据
void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());

}

/**
 * @brief   对imu和图像数据进行对齐并组合
 * @Description     img:    i -------- j  -  -------- k
 *                  imu:    - jjjjjjjj - j/k kkkkkkkk -  
 *                  直到把缓存中的图像特征数据或者IMU数据取完，才能够跳出此函数，并返回数据           
 * @return  vector<std::pair<vector<ImuConstPtr>, PointCloudConstPtr>> (IMUs, img_msg)s
*/
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (true)
    {
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;

        //对齐标准：IMU最后一个数据的时间要大于第一个图像特征数据的时间
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            //ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }

        //对齐标准：IMU第一个数据的时间要小于第一个图像特征数据的时间
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue; 
        }

        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        std::vector<sensor_msgs::ImuConstPtr> IMUs;

        //图像数据(img_msg)，对应多组在时间戳内的imu数据,然后塞入measurements
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            //emplace_back相比push_back能更好地避免内存的拷贝与移动
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();//弹出queue第一个元素
        }

        //这里把下一个imu_msg也放进去了,但没有pop，因此当前图像帧和下一图像帧会共用这个imu_msg
        IMUs.emplace_back(imu_buf.front());
        
        if (IMUs.empty())
            ROS_WARN("no imu between two image");

        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

//imu回调函数，将imu_msg保存到imu_buf，IMU状态递推并发布[P,Q,V,header]
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    //判断时间间隔是否为正
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        //ROS_WARN("imu message ins disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();

    // 将IMU数据存入IMU数据缓存队列imu_buf
    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
     //https://www.jianshu.com/p/c1dfa1d40f53 
    con.notify_one();//唤醒作用于process线程中的获取观测值数据的函数

    // 通过IMU测量值积分更新并发布里程计信息
    // 这一行代码似乎重复了，上面有着一模一样的代码
    last_imu_t = imu_msg->header.stamp.toSec();

    {
        //构造互斥锁m_state，析构时解锁
        std::lock_guard<std::mutex> lg(m_state);
        predict(imu_msg);//递推得到IMU的PQV
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";

        //发布最新的由IMU直接递推得到的PQV
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR) // VINS初始化已完成，正处于滑动窗口非线性优化状态，如果VINS还在初始化，则不发布里程计信息
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header); // 发布里程计信息，发布频率很高（与IMU数据同频），每次获取IMU数据都会及时进行更新，而且发布的是当前的里程计信息。
            // 还有一个pubOdometry()函数，似乎也是发布里程计信息，但是它是在estimator每次处理完一帧图像特征点数据后才发布的，有延迟，而且频率也不高（至多与图像同频）
    }
}

//feature回调函数，将feature_msg放入feature_buf     
void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }

    // 将图像特征点数据存入图像特征点数据缓存队列feature_buf
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();

 // 唤醒getMeasurements()读取缓存imu_buf和feature_buf中的观测数据
 //https://www.jianshu.com/p/c1dfa1d40f53
    con.notify_one();
}

//restart回调函数，收到restart时清空feature_buf和imu_buf，估计器重置，时间重置
void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");

        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();

        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();

        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

//relocalization回调函数，将points_msg放入relo_buf        
void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    //printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

/**
 * @brief   VIO的主线程
 * @Description 等待并获取measurements：(IMUs, img_msg)s，计算dt
 *        estimator.processIMU()进行IMU预积分         
 *        estimator.setReloFrame()设置重定位帧
 *        estimator.processImage()处理图像帧：初始化，紧耦合的非线性优化     
 * @return      void
*/
// process()是measurement_process线程的线程函数，
//在process()中处理VIO后端，包括IMU预积分、松耦合初始化和local BA
void process()
{
    while (true)
    {
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        
        std::unique_lock<std::mutex> lk(m_buf);

        //等待上面两个接收数据完成就会被唤醒
        //在提取measurements时互斥锁m_buf会锁住，此时无法接收数据
     // [&]{return (measurements = getMeasurements()).size() != 0;}是lamda表达式（匿名函数）
     // 先调用匿名函数，从缓存队列中读取IMU数据和图像特征点数据，如果measurements为空，则匿名函数返回false，调用wait(lock)，
     // 释放m_buf（为了使图像和IMU回调函数可以访问缓存队列），阻塞当前线程，等待被con.notify_one()唤醒
     // 直到measurements不为空时（成功从缓存队列获取数据），匿名函数返回true，则可以退出while循环。
        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0;
                 });
        lk.unlock();// 从缓存队列中读取数据完成，解锁

        m_estimator.lock();
        for (auto &measurement : measurements)
        {
            //对应这段的img data
            auto img_msg = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;

            //遍历该组中的各帧imu数据，进行预积分
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header.stamp.toSec();// 最新IMU数据的时间戳
                // 图像特征点数据的时间戳，补偿了通过优化得到的一个时间偏移
                double img_t = img_msg->header.stamp.toSec() + estimator.td;

                //发送IMU数据进行预积分
                // 补偿时间偏移后，图像特征点数据的时间戳晚于或等于IMU数据的时间戳
                if (t <= img_t)
                { 
                    if (current_time < 0) // 第一次接收IMU数据时会出现这种情况
                        current_time = t;
                    double dt = t - current_time;
                    ROS_ASSERT(dt >= 0);
                    current_time = t; // 更新最近一次接收的IMU数据的时间戳

                    // IMU数据测量值
                    // 3轴加速度测量值
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;

                    // 3轴角速度测量值
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;

                    //imu预积分
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

                }
                 // 时间戳晚于图像特征点数据（时间偏移补偿后）的第一帧IMU数据（也是一组measurement中的唯一一帧），
                 //对IMU数据进行插值，得到图像帧时间戳对应的IMU数据
                else
                {
                     // 时间戳的对应关系如下图所示：
                    //                                            current_time         t
                    // *               *               *               *               *     （IMU数据）
                    //                                                          |            （图像特征点数据）
                    //                                                        img_t
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);//这里的权重没看懂
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }
        
            // 设置重定位用的回环帧
            // set relocalization frame
            sensor_msgs::PointCloudConstPtr relo_msg = NULL;
            
            //取出最后一个重定位帧
            while (!relo_buf.empty())
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }

            if (relo_msg != NULL)
            {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec(); // 回环帧的时间戳
                for (unsigned int i = 0; i < relo_msg->points.size(); i++)
                {
                    Vector3d u_v_id;
                    u_v_id.x() = relo_msg->points[i].x;
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id);
                }
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                frame_index = relo_msg->channels[0].values[7];

                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
            }

            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

            TicToc t_s;

            //建立每个特征点的(camera_id,[x,y,z,u,v,vx,vy])s的map，索引为feature_id
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                int v = img_msg->channels[0].values[i] + 0.5;
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                double p_u = img_msg->channels[1].values[i];
                double p_v = img_msg->channels[2].values[i];
                double velocity_x = img_msg->channels[3].values[i];
                double velocity_y = img_msg->channels[4].values[i];
                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
            }
            
            //处理图像特征
            estimator.processImage(image, img_msg->header);

            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";

            //给RVIZ发送topic
            pubOdometry(estimator, header);//"odometry" 里程计信息PQV
            pubKeyPoses(estimator, header);//"key_poses" 关键点三维坐标
            pubCameraPose(estimator, header);//"camera_pose" 相机位姿
            pubPointCloud(estimator, header);//"history_cloud" 点云信息
            pubTF(estimator, header);//"extrinsic" 相机到IMU的外参
            pubKeyframe(estimator);//"keyframe_point"、"keyframe_pose" 关键帧位姿和点云

            if (relo_msg != NULL)
                pubRelocalization(estimator);//"relo_relative_pose" 重定位位姿
            //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        m_estimator.unlock();

        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
         // VINS系统完成滑动窗口优化后，用优化后的结果，更新里程计数据
            update();//更新IMU参数[P,Q,V,ba,bg,a,g]
        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char **argv)
{
    //ROS初始化，设置句柄n
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    
    //读取参数，设置估计器参数
    readParameters(n);
    estimator.setParameter();

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");

    //用于RVIZ显示的Topic
    registerPub(n);// 注册visualization.cpp中创建的发布器

    //订阅IMU、feature、restart、match_points的topic,执行各自回调函数
     // IMU数据
    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
   // 图像特征点数据
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    // ？？？接收一个bool值，判断是否重启estimator
    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);
    // ？？？根据回环检测信息进行重定位
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);

    //创建VIO主线程
    // estimator_node中的线程由旧版本中的3个改为现在的1个
    // 回环检测和全局位姿图优化在新增的一个ROS节点中运行
    // measurement_process线程的线程函数是process()，在process()中处理VIO后端，包括IMU预积分、松耦合初始化和local BA
    std::thread measurement_process{process};
    ros::spin();

    return 0;
}

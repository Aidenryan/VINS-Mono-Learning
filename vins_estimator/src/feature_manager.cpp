#include "feature_manager.h"

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();//setIdentity() 矩阵自带的初始化函数 初始化为单位矩阵
}

void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];//
    }
}

//清楚特征管理器中的特征
void FeatureManager::clearState()
{
    feature.clear();
}

//窗口中被跟踪的角点数量
int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature)
    {

        it.used_num = it.feature_per_frame.size();

        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
        {
            cnt++;
        }
    }
    return cnt;
}

/**
 * @brief   把特征点放入feature的list容器中，计算每一个点跟踪次数和它在次新帧和次次新帧间的视差，返回是否是关键帧
 * @param[in]   frame_count 窗口内帧的个数
 * @param[in]   image 某帧所有特征点的[camera_id,[x,y,z,u,v,vx,vy]]s构成的map,索引为feature_id
 * @param[in]   td IMU和cam同步时间差
 * @return  bool true：次新帧是关键帧;false：非关键帧
*/
/**
 * 把当前帧图像（frame_count）的特征点添加到feature容器中
 * 计算第2最新帧与第3最新帧之间的平均视差（当前帧是第1最新帧）
 * 也就是说当前帧图像特征点存入feature中后，并不会立即判断是否将当前帧添加为新的关键帧，而是去判断当前帧的前一帧（第2最新帧）。
 * 当前帧图像要在下一次接收到图像时进行判断（那个时候，当前帧已经变成了第2最新帧）
 */
//共三个判断条件
bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0; // 第2最新帧和第3最新帧之间跟踪到的特征点的总视差
    int parallax_num = 0; // 第2最新帧和第3最新帧之间跟踪到的特征点的数量
    last_track_num = 0; // 当前帧（第1最新帧）图像跟踪到的特征点的数量

     // 把当前帧图像特征点数据image添加到feature容器中
    // feature容器按照特征点id组织特征点数据，对于每个id的特征点，记录它被滑动窗口中哪些图像帧观测到了
    for (auto &id_pts : image)
    {
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);//f_per_fra 记录该特征点在该图像的信息

        //迭代器寻找feature list中是否有这feature_id
        int feature_id = id_pts.first;


         /**
         * STL find_if的用法：
         * find_if (begin, end, func)
         * 就是从begin开始 ，到end为止，返回第一个让 func这个函数返回true的iterator
         */
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          {
            return it.feature_id == feature_id;
                          });

        // 返回尾部迭代器，说明该特征点第一次出现（在当前帧中新检测的特征点），需要在feature中新建一个FeaturePerId对象
        //如果没有则新建一个，并添加这图像帧
        if (it == feature.end())
        {
            feature.push_back(FeaturePerId(feature_id, frame_count));
            feature.back().feature_per_frame.push_back(f_per_fra);
        }
        //有的话把图像帧添加进去
        else if (it->feature_id == feature_id)
        {
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++;
        }
    }

    // 1. 当前帧的帧号小于2，即为0或1，为0，则没有第2最新帧，为1，则第2最新帧是滑动窗口中的第1帧
    // 2. 当前帧（第1最新帧）跟踪到的特征点数量小于20（？？？为什么当前帧的跟踪质量不好，就把第2最新帧当作关键帧？？？）
    // 出现以上2种情况的任意一种，则认为第2最新帧是关键帧
    if (frame_count < 2 || last_track_num < 20)
        return true;// 第2最新帧是关键帧

    //计算每个特征在次新帧和次次新帧中的视差
    // 计算第2最新帧和第3最新帧之间跟踪到的特征点的平均视差
    for (auto &it_per_id : feature)
    {
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)//后面那个条件没看懂
        {
            // 对于给定id的特征点，计算第2最新帧和第3最新帧之间该特征点的视差（当前帧frame_count是第1最新帧）
            //（需要使用IMU数据补偿由于旋转造成的视差）
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }

    if (parallax_num == 0)
    {
        // 如果第2最新帧和第3最新帧之间跟踪到的特征点的数量为0，则把第2最新帧添加为关键帧
        // ？？怎么会出现这种情况？？？？
        // 如果出现这种情况，那么第2最新帧和第3最新帧之间的视觉约束关系不就没有了？？？
        return true;
    }
    else
    {
        // 计算平均视差
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);//FOCAL_LENGTH焦距
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

void FeatureManager::debugShow()
{
    ROS_DEBUG("debug show");
    for (auto &it : feature)
    {
        ROS_ASSERT(it.feature_per_frame.size() != 0);
        ROS_ASSERT(it.start_frame >= 0);
        ROS_ASSERT(it.used_num >= 0);

        ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto &j : it.feature_per_frame)
        {
            ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ",j.point(0), j.point(1));
        }
        ROS_ASSERT(it.used_num == sum);
    }
}

//得到frame_count_l与frame_count_r两帧之间的对应特征点
//没看懂这个函数干啥
vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature)
    {
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;
            
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

//设置特征点的逆深度估计值
void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        //如果该角点出现的帧数不超过2或者第一次出现的帧在倒数三个窗口内（最近才出现）
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);//??
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;//失败估计
        }
        else
            it_per_id.solve_flag = 1;//成功估计
    }
}

//剔除feature中估计失败的点（solve_flag == 2）
void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            feature.erase(it);//clear清除所有元素，erase清除某个元素
    }
}

void FeatureManager::clearDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
    }
}


VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

//对特征点进行三角化求深度（SVD分解）
void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    for (auto &it_per_id : feature)// 对于每个id的特征点
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();// 每个id的特征点被多少帧图像观测到了
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            
            // 如果该特征点被两帧及两帧以上的图像观测到，
            // 且观测到该特征点的第一帧图像应该早于或等于滑动窗口第4最新关键帧
            // 也就是说，至少是第4最新关键帧和第3最新关键帧观测到了该特征点（第2最新帧似乎是紧耦合优化的最新帧）    
            continue;

            // 该id的特征点深度值大于0，该值在初始化时为-1，如果大于0，说明该点被三角化过
        if (it_per_id.estimated_depth > 0)
            continue;

        // imu_i：观测到该特征点的第一帧图像在滑动窗口中的帧号
        // imu_j：观测到该特征点的最后一帧图像在滑动窗口中的帧号
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        ROS_ASSERT(NUM_OF_CAM == 1);
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        //R0 t0为第i帧相机坐标系到世界坐标系的变换矩阵
        Eigen::Matrix<double, 3, 4> P0;// 似乎是[R | T]的形式，是一个位姿
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
        P0.leftCols<3>() = Eigen::Matrix3d::Identity(); // 单位旋转矩阵
        P0.rightCols<1>() = Eigen::Vector3d::Zero(); // 0平移向量

        for (auto &it_per_frame : it_per_id.feature_per_frame)// 对于观测到该id特征点的每一图像帧
        {
            imu_j++;// 观测到该特征点的最后一帧图像在滑动窗口中的帧号


            //R t为第j帧相机坐标系到第i帧相机坐标系的变换矩阵，P为i到j的变换矩阵
            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Eigen::Vector3d f = it_per_frame.point.normalized();
            //P = [P1 P2 P3]^T 
            //AX=0      A = [A(2*i) A(2*i+1) A(2*i+2) A(2*i+3) ...]^T
            //A(2*i)   = x(i) * P3 - z(i) * P1
            //A(2*i+1) = y(i) * P3 - z(i) * P2
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j)// 在第一次进入for循环的时候，这个条件成立，这时候循环体都执行完了，continue发挥不了什么作用啊？？？
                continue;
        }

        //对A的SVD分解得到其最小奇异值对应的单位奇异向量(x,y,z,w)，深度为z/w
        ROS_ASSERT(svd_idx == svd_A.rows());
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];
        //it_per_id->estimated_depth = -b / A;
        //it_per_id->estimated_depth = svd_V[2] / svd_V[3];

        it_per_id.estimated_depth = svd_method;// 似乎是得到了该特征点的深度
        //it_per_id->estimated_depth = INIT_DEPTH;

        if (it_per_id.estimated_depth < 0.1)// 如果估计出来的深度小于0.1（单位是啥？？？），则把它替换为一个设定的值
        {
            it_per_id.estimated_depth = INIT_DEPTH;
        }

    }
}

//移除外点
void FeatureManager::removeOutlier()
{
    ROS_BREAK();
    int i = -1;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true)
        {
            feature.erase(it);
        }
    }
}

//边缘化最老帧时，处理特征点保存的帧号，将起始帧是最老帧的特征点的深度值进行转移
//marg_R、marg_P为被边缘化的位姿，new_R、new_P为在这下一帧的位姿
void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        //特征点起始帧不是最老帧则将帧号减一
        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            //特征点起始帧是最老帧
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;  //[0]可以这种吗
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            //特征点只在最老帧被观测则直接移除
            if (it->feature_per_frame.size() < 2)
            {
                feature.erase(it);
                continue;
            }
            else
            {
                //pts_i为特征点在最老帧坐标系下的三维坐标
                //w_pts_i为特征点在世界坐标系下的三维坐标
                //将特征点转换到在下一帧坐标系下的坐标 pts_j
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);//不应该是取逆吗 怎么是转置
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;//深度初始值
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}

//边缘化最老帧时，直接将特征点所保存的帧号向前滑动
void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        //如果特征点起始帧号start_frame不为零则减一
        if (it->start_frame != 0)
            it->start_frame--;
        //如果start_frame为0则直接移除feature_per_frame的第0帧FeaturePerFrame
        //如果feature_per_frame为空则直接删除特征点
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

//边缘化次新帧时，对特征点在次新帧的信息进行移除处理
void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;
        //起始帧为最新帧的滑动成次新帧
        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            //如果次新帧之前已经跟踪结束则什么都不做
            if (it->endFrame() < frame_count - 1)
                continue;
            //如果在次新帧仍被跟踪，则删除feature_per_frame中次新帧对应的FeaturePerFrame
            //如果feature_per_frame为空则直接删除特征点
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

//计算某个特征点it_per_id在次新帧和次次新帧的视差

/**
 * 对于给定id的特征点
 * 计算第2最新帧和第3最新帧之间该特征点的视差（当前帧frame_count是第1最新帧）
 * （需要使用IMU数据补偿由于旋转造成的视差）
 */
double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0;// 初始化视差
    Vector3d p_j = frame_j.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    //  不用比也是后者大啊？？？？
    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}
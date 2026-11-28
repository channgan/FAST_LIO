// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <cstring>
#include <chrono>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include "flight_odometry.hpp"
#include "fast_lio/msg/frame_timing.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include "preprocess.h"
#include "adaptive_scan.hpp"
#include "constraint_selection.hpp"
#include <stdexcept>
#include <ikd-Tree/ikd_Tree.h>
#include "plane_map.hpp"

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

vector<float> res_last;
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
vector<uint8_t> point_selected_surf;
bool   flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
bool   adaptive_scan_pub_en = false;
bool   adaptive_frontend = false;
fast_lio::AdaptiveScanOptions scan_options;
fast_lio::AdaptiveScanResult adaptive_scan;
PointCloudXYZI::Ptr adaptive_representatives(new PointCloudXYZI());
fast_lio::SelectionMode selection_mode = fast_lio::SelectionMode::All;
int match_budget = 600, candidate_count = 0;
double rotation_length = 5.0;
fast_lio::SelectionStats selection_stats;
std::ofstream frame_metrics;
using FrameTiming = fast_lio::msg::FrameTiming;
rclcpp::Publisher<FrameTiming>::SharedPtr timing_pub;

void record_timing(int64_t end, uint8_t stage, int64_t steady)
{
    if (!timing_pub->get_subscription_count()) return;
    FrameTiming event;
    event.scan_end_ns = end;
    event.stage = stage;
    event.steady_ns = steady;
    timing_pub->publish(event);
}
bool    is_first_lidar = true;

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
struct BufferedScan
{
    PointCloudXYZI::Ptr points;
    rclcpp::Time begin;
    rclcpp::Time end;
    double preprocess_ms = 0.0;
};
deque<BufferedScan> lidar_buffer;
deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI());
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI());
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI());
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;
bool plane_association = false;
fast_lio::LocalPlaneMap plane_map;
std::size_t matched_plane_count = 0;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::msg::Path path;
geometry_msgs::msg::Quaternion geoQuat;
geometry_msgs::msg::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

void SigHandle(int sig)
{
    flg_exit = true;
    std::cout << "catch sig %d" << sig << std::endl;
    sig_buffer.notify_all();
    rclcpp::shutdown();
}

inline void dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a  
    fprintf(fp, "\r\n");  
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
void update_plane_window()
{
    Eigen::Vector3d lower, upper;
    for (int i = 0; i < 3; ++i) {
        lower[i] = LocalMap_Points.vertex_min[i];
        upper[i] = LocalMap_Points.vertex_max[i];
    }
    plane_map.setWindow(lower, upper);
}

bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;    
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        if (plane_association) update_plane_window();
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;
    if (plane_association) {
        update_plane_window();
        return;
    }

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

// A Livox packet carries nanosecond offsets even for points removed by
// preprocessing. Other drivers may only expose relative times through curvature.
void buffer_scan(const PointCloudXYZI::Ptr& points, const rclcpp::Time& begin,
                 int64_t duration_ns = -1)
{
    static double mean_duration_ns = 0.0;
    static size_t duration_count = 0;
    bool measured = duration_ns >= 0;
    if (!measured && points->size() > 1)
    {
        float max_offset_ms = 0.0f;
        for (const auto& point : points->points)
            max_offset_ms = std::max(max_offset_ms, point.curvature);
        duration_ns = std::llround(max_offset_ms * 1e6);
        measured = duration_ns >= 0.5 * mean_duration_ns;
    }
    if (measured)
        mean_duration_ns += (duration_ns - mean_duration_ns) / ++duration_count;
    else
        duration_ns = std::llround(mean_duration_ns);
    lidar_buffer.push_back({points, begin, begin + rclcpp::Duration::from_nanoseconds(duration_ns)});
}

void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::UniquePtr msg) 
{
    const auto entered = fast_lio::steady_ns();
    mtx_buffer.lock();
    scan_count ++;
    double cur_time = get_time_sec(msg->header.stamp);
    double preprocess_start_time = omp_get_wtime();
    if (!is_first_lidar && cur_time < last_timestamp_lidar)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        lidar_buffer.clear();
    }
    if (is_first_lidar)
    {
        is_first_lidar = false;
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    buffer_scan(ptr, rclcpp::Time(msg->header.stamp));
    last_timestamp_lidar = cur_time;
    lidar_buffer.back().preprocess_ms = (omp_get_wtime() - preprocess_start_time) * 1000;
    if (runtime_pos_log && scan_count < MAXN) s_plot11[scan_count] = lidar_buffer.back().preprocess_ms / 1000;
    const auto end = lidar_buffer.back().end.nanoseconds();
    const auto ready = fast_lio::steady_ns();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    record_timing(end, FrameTiming::CALLBACK, entered);
    record_timing(end, FrameTiming::PREPROCESSED, ready);
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::UniquePtr msg) 
{
    const auto entered = fast_lio::steady_ns();
    mtx_buffer.lock();
    double cur_time = get_time_sec(msg->header.stamp);
    double preprocess_start_time = omp_get_wtime();
    scan_count ++;
    if (!is_first_lidar && cur_time < last_timestamp_lidar)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        lidar_buffer.clear();
    }
    if(is_first_lidar)
    {
        is_first_lidar = false;
    }
    last_timestamp_lidar = cur_time;
    
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    int64_t duration_ns = -1;
    for (const auto& point : msg->points)
        duration_ns = std::max(duration_ns, static_cast<int64_t>(point.offset_time));
    buffer_scan(ptr, rclcpp::Time(msg->header.stamp), duration_ns);
    
    lidar_buffer.back().preprocess_ms = (omp_get_wtime() - preprocess_start_time) * 1000;
    if (runtime_pos_log && scan_count < MAXN) s_plot11[scan_count] = lidar_buffer.back().preprocess_ms / 1000;
    const auto end = lidar_buffer.back().end.nanoseconds();
    const auto ready = fast_lio::steady_ns();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    record_timing(end, FrameTiming::CALLBACK, entered);
    record_timing(end, FrameTiming::PREPROCESSED, ready);
}

void buffer_imu(sensor_msgs::msg::Imu::UniquePtr input)
{
    sensor_msgs::msg::Imu::SharedPtr msg(std::move(input));
    std::lock_guard<std::mutex> lock(mtx_buffer);
    const double time_offset = time_sync_en && std::abs(timediff_lidar_wrt_imu) > 0.1
        ? timediff_lidar_wrt_imu : -time_diff_lidar_to_imu;
    msg->header.stamp = rclcpp::Time(msg->header.stamp) +
        rclcpp::Duration::from_nanoseconds(std::llround(time_offset * 1e9));

    double timestamp = get_time_sec(msg->header.stamp);

    if (timestamp < last_timestamp_imu)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    sig_buffer.notify_all();
}

bool sync_packages(MeasureGroup &meas)
{
    std::lock_guard<std::mutex> lock(mtx_buffer);
    if (lidar_buffer.empty())
        return false;
    const auto& scan = lidar_buffer.front();
    if (!fast_lio::collect_frame_imu(imu_buffer, scan.end, meas.imu, meas.imu_after_end))
        return false;
    meas.lidar = scan.points;
    meas.preprocess_ms = scan.preprocess_ms;
    meas.lidar_begin_stamp = scan.begin;
    meas.lidar_end_stamp = scan.end;
    meas.lidar_beg_time = scan.begin.seconds();
    lidar_buffer.pop_front();
    return true;
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI());
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        const auto* source_points = &feats_down_body->points;
        if (dense_pub_en)
        {
            source_points = &feats_undistort->points;
        }
        else if (adaptive_scan_pub_en)
        {
            source_points = &adaptive_representatives->points;
        }

        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(source_points->size(), 1));
        for (std::size_t i = 0; i < source_points->size(); ++i)
        {
            RGBpointBodyToWorld(&(*source_points)[i], &laserCloudWorld->points[i]);
        }

        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = Measures.lidar_end_stamp;
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull->publish(laserCloudmsg);
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    /*
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
    */
}

void publish_frame_body(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& full_pub,
                        const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& xyz_pub)
{
    const bool want_full = full_pub->get_subscription_count() > 0;
    const bool want_xyz = xyz_pub->get_subscription_count() > 0;
    if (!want_full && !want_xyz) return;
    const auto count = feats_undistort->points.size();
    PointCloudXYZI full_cloud;
    if (want_full) {
        full_cloud.resize(count);
        full_cloud.width = count;
        full_cloud.height = 1;
    }
    sensor_msgs::msg::PointCloud2 compact;
    if (want_xyz) {
        compact.height = 1;
        compact.is_dense = feats_undistort->is_dense;
        const uint16_t endian_probe = 1;
        compact.is_bigendian = (*reinterpret_cast<const uint8_t*>(&endian_probe) == 0);
        sensor_msgs::PointCloud2Modifier fields(compact);
        const auto type = sensor_msgs::msg::PointField::FLOAT32;
        fields.setPointCloud2Fields(3, "x", 1, type, "y", 1, type, "z", 1, type);
        fields.resize(count);
    }
    for (std::size_t i = 0; i < count; ++i) {
        PointType scratch;
        PointType& point = want_full ? full_cloud.points[i] : scratch;
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], &point);
        if (want_xyz)
            std::memcpy(compact.data.data() + i * compact.point_step, point.data, 3 * sizeof(float));
    }
    std_msgs::msg::Header header;
    header.stamp = Measures.lidar_end_stamp;
    header.frame_id = "body";
    if (want_xyz) {
        compact.header = header;
        xyz_pub->publish(compact);
    }
    if (want_full) {
        sensor_msgs::msg::PointCloud2 full;
        pcl::toROSMsg(full_cloud, full);
        full.header = header;
        full_pub->publish(full);
    }
}

void publish_effect_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = Measures.lidar_end_stamp;
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect->publish(laserCloudFullRes3);
}

void snapshot_plane_map()
{
    pcl_wait_pub->clear();
    pcl_wait_pub->reserve(plane_map.size());
    plane_map.visit([](const fast_lio::MapPlane& plane) {
        PointType point{};
        point.x = plane.points.mean.x();
        point.y = plane.points.mean.y();
        point.z = plane.points.mean.z();
        point.normal_x = plane.normal.x();
        point.normal_y = plane.normal.y();
        point.normal_z = plane.normal.z();
        pcl_wait_pub->push_back(point);
    });
}

void publish_map(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap)
{
    if (plane_association) {
        snapshot_plane_map();
    } else {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_pub += *laserCloudWorld;

    }

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*pcl_wait_pub, laserCloudmsg);
    laserCloudmsg.header.stamp = Measures.lidar_end_stamp;
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserCloudMap->publish(laserCloudmsg);

    // sensor_msgs::msg::PointCloud2 laserCloudMap;
    // pcl::toROSMsg(*featsFromMap, laserCloudMap);
    // laserCloudMap.header.stamp = Measures.lidar_end_stamp;
    // laserCloudMap.header.frame_id = "camera_init";
    // pubLaserCloudMap->publish(laserCloudMap);
}

void save_to_pcd()
{
    if (plane_association) snapshot_plane_map();
    pcl::PCDWriter pcd_writer;
    pcd_writer.writeBinary(map_file_path, *pcl_wait_pub);
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
    
}

void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr& pubOdomAftMapped,
                      std::unique_ptr<tf2_ros::TransformBroadcaster>& tf_br,
                      fast_lio::FlightOdometry* external_observation)
{
    nav_msgs::msg::Odometry odomAftMapped;
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = Measures.lidar_end_stamp;
    set_posestamp(odomAftMapped.pose);
    // Position errors are in camera_init; the filter's right-multiplicative rotation errors are in body.
    Eigen::Matrix<double, 6, 6> to_fixed_axes = Eigen::Matrix<double, 6, 6>::Identity();
    to_fixed_axes.bottomRightCorner<3, 3>() = state_point.rot.toRotationMatrix();
    Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> pose_covariance(
        odomAftMapped.pose.covariance.data());
    pose_covariance = to_fixed_axes * kf.get_P().topLeftCorner<6, 6>() * to_fixed_axes.transpose();
    const Eigen::Vector3d body_velocity = state_point.rot.toRotationMatrix().transpose() * state_point.vel;
    auto& twist = odomAftMapped.twist.twist;
    twist.linear.x = body_velocity.x();
    twist.linear.y = body_velocity.y();
    twist.linear.z = body_velocity.z();
    // The deskewer already interpolated this measurement at the scan end.
    const auto& imu = p_imu->frame_end_sample();
    if (imu && rclcpp::Time(imu->header.stamp) == Measures.lidar_end_stamp) {
        const auto& gyro = imu->angular_velocity;
        twist.angular.x = gyro.x - state_point.bg.x();
        twist.angular.y = gyro.y - state_point.bg.y();
        twist.angular.z = gyro.z - state_point.bg.z();
        if (external_observation) external_observation->submit(odomAftMapped);
    }
    pubOdomAftMapped->publish(odomAftMapped);

    geometry_msgs::msg::TransformStamped trans;
    trans.header.frame_id = "camera_init";
    trans.child_frame_id = "body";
    trans.header.stamp = Measures.lidar_end_stamp;
    trans.transform.translation.x = odomAftMapped.pose.pose.position.x;
    trans.transform.translation.y = odomAftMapped.pose.pose.position.y;
    trans.transform.translation.z = odomAftMapped.pose.pose.position.z;
    trans.transform.rotation.w = odomAftMapped.pose.pose.orientation.w;
    trans.transform.rotation.x = odomAftMapped.pose.pose.orientation.x;
    trans.transform.rotation.y = odomAftMapped.pose.pose.orientation.y;
    trans.transform.rotation.z = odomAftMapped.pose.pose.orientation.z;
    tf_br->sendTransform(trans);
}

void publish_path(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = Measures.lidar_end_stamp;
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path.poses.push_back(msg_body_pose);
        pubPath->publish(path);
    }
}

Eigen::Vector3d scan_position(std::size_t index)
{
    const auto& point = feats_undistort->points[index];
    return Eigen::Vector3d(point.x, point.y, point.z);
}

Eigen::Matrix3d lidar_rotation(const state_ikfom& state)
{
    return state.rot.toRotationMatrix() * state.offset_R_L_I.toRotationMatrix();
}

void integrate_plane_scan()
{
    plane_map.integrate(adaptive_scan.patches, scan_position, lidar_rotation(state_point), pos_lid);
}

// Choose separated observed returns (four tangent extrema) to retain each
// patch's rotation lever arms without fusing every densely sampled hit.
void collect_plane_observations(const state_ikfom& state)
{
    const Eigen::Matrix3d rotation = lidar_rotation(state);
    const Eigen::Vector3d translation = state.pos + state.rot * state.offset_T_L_I;
    const auto pairs = plane_map.associate(adaptive_scan.patches, rotation, translation);
    matched_plane_count = 0;
    for (const auto& pair : pairs) {
        const auto& patch = adaptive_scan.patches[pair.scan_index];
        const auto& plane = *pair.target;
        const Eigen::Vector3d u = patch.normal.unitOrthogonal(), v = patch.normal.cross(u);
        std::array<double, 4> score;
        score.fill(-std::numeric_limits<double>::infinity());
        std::array<std::size_t, 4> supports{};
        for (auto index : patch.support_indices) {
            const Eigen::Vector3d p = scan_position(index);
            const Eigen::Vector3d world = rotation * p + translation;
            const Eigen::Vector3d gap = (world - plane.points.upper).cwiseMax(plane.points.lower - world).cwiseMax(0.);
            if (gap.norm() > filter_size_map_min * 0.35) continue;
            const double a = u.dot(p), b = v.dot(p);
            const std::array<double, 4> projection{{a, -a, b, -b}};
            for (int k = 0; k < 4; ++k) {
                if (projection[k] > score[k]) {
                    score[k] = projection[k];
                    supports[k] = index;
                }
            }
        }
        if (!std::isfinite(score[0])) continue;
        std::sort(supports.begin(), supports.end());
        const auto end = std::unique(supports.begin(), supports.end());
        const auto before = laserCloudOri->size();
        for (auto it = supports.begin(); it != end; ++it) {
            const Eigen::Vector3d p = scan_position(*it);
            const double residual = plane.normal.dot(rotation * p + translation) + plane.offset;
            // Same point-to-plane gate as the nearest-neighbor reference path.
            if (std::abs(residual) >= std::sqrt(p.norm()) / 9.0) continue;
            PointType measurement{};
            measurement.x = plane.normal.x();
            measurement.y = plane.normal.y();
            measurement.z = plane.normal.z();
            measurement.intensity = residual;
            laserCloudOri->push_back(feats_undistort->points[*it]);
            corr_normvect->push_back(measurement);
            total_residual += std::abs(residual);
        }
        if (laserCloudOri->size() != before) ++matched_plane_count;
    }
}

void collect_knn_observations(state_ikfom& s, bool rematch)
{
    /** closest surface search and residual computation **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i]; 
        PointType &point_world = feats_down_world->points[i]; 

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (rematch)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }


    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->push_back(feats_down_body->points[i]);
            corr_normvect->push_back(normvec->points[i]);
            total_residual += res_last[i];
        }
    }

}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear();
    corr_normvect->clear();
    total_residual = 0.0;

    if (plane_association)
        collect_plane_observations(s);
    else
        collect_knn_observations(s, ekfom_data.converge);
    effct_feat_num = laserCloudOri->size();
    candidate_count = effct_feat_num;
    selection_stats = {};
    res_mean_last = 0.0;

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        std::cerr << "No Effective Points!" << std::endl;
        // ROS_WARN("No Effective Points! \n");
        return;
    }

    res_mean_last = total_residual / effct_feat_num;
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
    fast_lio::MatchPositions positions;
    if (selection_mode != fast_lio::SelectionMode::All && effct_feat_num > match_budget)
    {
        positions.resize(effct_feat_num, 3);
        for (int i = 0; i < effct_feat_num; ++i)
            positions.row(i) = laserCloudOri->points[i].getVector3fMap().cast<double>().transpose();
    }
    const auto selected = fast_lio::selectMatches(ekfom_data.h_x, positions, selection_mode,
        match_budget, rotation_length, frame_metrics.is_open() ? &selection_stats : nullptr);
    if (!selected.empty())
    {
        effct_feat_num = selected.size();
        for (int i = 0; i < effct_feat_num; ++i)
        {
            const int src = selected[i];
            ekfom_data.h_x.row(i) = ekfom_data.h_x.row(src);
            ekfom_data.h(i) = ekfom_data.h(src);
            laserCloudOri->points[i] = laserCloudOri->points[src];
            corr_normvect->points[i] = corr_normvect->points[src];
        }
        ekfom_data.h_x.conservativeResize(effct_feat_num, Eigen::NoChange);
        ekfom_data.h.conservativeResize(effct_feat_num);
        laserCloudOri->resize(effct_feat_num);
        corr_normvect->resize(effct_feat_num);
    }
    solve_time += omp_get_wtime() - solve_start_;
}

void write_frame_metrics(const char* status, double frame_start, double map_seconds)
{
    if (!frame_metrics.is_open()) return;
    const auto& q = state_point.offset_R_L_I.coeffs();
    frame_metrics << Measures.lidar_end_stamp.nanoseconds() << ',' << status << ',' << feats_undistort->size()
        << ',' << feats_down_size << ',' << candidate_count << ',' << effct_feat_num
        << ',' << selection_stats.weakest_all << ',' << selection_stats.weakest_selected
        << ',' << adaptive_scan.patches.size() << ',' << matched_plane_count << ',' << plane_map.size()
        << ',' << map_seconds * 1000 << ',' << Measures.preprocess_ms
        << ',' << Measures.preprocess_ms + (omp_get_wtime() - frame_start) * 1000 << ',' << res_mean_last
        << ',' << state_point.pos.x()
        << ',' << state_point.pos.y() << ',' << state_point.pos.z();
    for (int i=0; i<3; ++i) frame_metrics << ',' << state_point.offset_T_L_I[i];
    for (int i=0; i<4; ++i) frame_metrics << ',' << q[i];
    frame_metrics << ',';
    for (const auto& group : adaptive_scan.normal_groups)
        frame_metrics << group.normal.x() << ':' << group.normal.y() << ':' << group.normal.z()
                      << '=' << group.before << '/' << group.after << ' ';
    frame_metrics << std::endl;
}

class LaserMappingNode : public rclcpp::Node
{
public:
    LaserMappingNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions()) : Node("laser_mapping", options)
    {
        this->declare_parameter<bool>("publish.path_en", true);
        this->declare_parameter<bool>("publish.effect_map_en", false);
        this->declare_parameter<bool>("publish.map_en", false);
        this->declare_parameter<bool>("publish.scan_publish_en", true);
        this->declare_parameter<bool>("publish.dense_publish_en", true);
        adaptive_scan_pub_en = this->declare_parameter<bool>("publish.adaptive_scan_en", false);
        this->declare_parameter<bool>("publish.scan_bodyframe_pub_en", true);
        this->declare_parameter<int>("max_iteration", 4);
        this->declare_parameter<string>("map_file_path", "");
        this->declare_parameter<string>("common.lid_topic", "/livox/lidar");
        this->declare_parameter<string>("common.imu_topic", "/livox/imu");
        this->declare_parameter<bool>("common.time_sync_en", false);
        this->declare_parameter<double>("common.time_offset_lidar_to_imu", 0.0);
        this->declare_parameter<double>("filter_size_corner", 0.5);
        this->declare_parameter<double>("filter_size_surf", 0.5);
        this->declare_parameter<double>("filter_size_map", 0.5);
        this->declare_parameter<double>("cube_side_length", 200.);
        this->declare_parameter<float>("mapping.det_range", 300.);
        const auto association = this->declare_parameter<string>("mapping.association", "planes");
        if (association != "knn" && association != "planes")
            throw std::invalid_argument("mapping.association must be knn or planes");
        plane_association = association == "planes";
        this->declare_parameter<double>("mapping.fov_degree", 180.);
        this->declare_parameter<double>("mapping.gyr_cov", 0.1);
        this->declare_parameter<double>("mapping.acc_cov", 0.1);
        this->declare_parameter<double>("mapping.b_gyr_cov", 0.0001);
        this->declare_parameter<double>("mapping.b_acc_cov", 0.0001);
        this->declare_parameter<double>("preprocess.blind", 0.01);
        this->declare_parameter<int>("preprocess.lidar_type", AVIA);
        this->declare_parameter<int>("preprocess.scan_line", 16);
        this->declare_parameter<int>("preprocess.timestamp_unit", US);
        this->declare_parameter<int>("preprocess.scan_rate", 10);
        this->declare_parameter<int>("point_filter_num", 2);
        this->declare_parameter<bool>("feature_extract_enable", false);
        this->declare_parameter<bool>("runtime_pos_log_enable", false);
        this->declare_parameter<bool>("mapping.extrinsic_est_en", true);
        this->declare_parameter<bool>("pcd_save.pcd_save_en", false);
        this->declare_parameter<int>("pcd_save.interval", -1);
        this->declare_parameter<vector<double>>("mapping.extrinsic_T", vector<double>());
        this->declare_parameter<vector<double>>("mapping.extrinsic_R", vector<double>());

        this->get_parameter_or<bool>("publish.path_en", path_en, true);
        this->get_parameter_or<bool>("publish.effect_map_en", effect_pub_en, false);
        this->get_parameter_or<bool>("publish.map_en", map_pub_en, false);
        this->get_parameter_or<bool>("publish.scan_publish_en", scan_pub_en, true);
        this->get_parameter_or<bool>("publish.dense_publish_en", dense_pub_en, true);
        this->get_parameter_or<bool>("publish.scan_bodyframe_pub_en", scan_body_pub_en, true);
        this->get_parameter_or<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
        this->get_parameter_or<string>("map_file_path", map_file_path, "");
        this->get_parameter_or<string>("common.lid_topic", lid_topic, "/livox/lidar");
        this->get_parameter_or<string>("common.imu_topic", imu_topic,"/livox/imu");
        this->get_parameter_or<bool>("common.time_sync_en", time_sync_en, false);
        this->get_parameter_or<double>("common.time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
        this->get_parameter_or<double>("filter_size_corner",filter_size_corner_min,0.5);
        this->get_parameter_or<double>("filter_size_surf",filter_size_surf_min,0.5);
        adaptive_frontend = this->declare_parameter<bool>("mapping.adaptive_scan_en", false);
        scan_options.leaf_size = filter_size_surf_min;
        scan_options.min_plane_points = this->declare_parameter<int>("planes.min_points", 6);
        scan_options.min_spread_ratio = this->declare_parameter<double>("planes.min_spread_ratio", 0.04);
        scan_options.max_flatness_ratio = this->declare_parameter<double>("planes.max_flatness_ratio", 0.05);
        scan_options.max_residual_ratio = this->declare_parameter<double>("planes.max_residual_ratio", 0.05);
        scan_options.normal_angle_deg = this->declare_parameter<double>("planes.normal_angle_deg", 15.);
        scan_options.normal_group_cap = this->declare_parameter<int>("planes.normal_group_cap", 0);
        const auto selection = this->declare_parameter<string>("mapping.match_selection", "all");
        if (selection == "constraint") selection_mode = fast_lio::SelectionMode::Constraint;
        else if (selection == "uniform") selection_mode = fast_lio::SelectionMode::Uniform;
        else if (selection != "all") throw std::invalid_argument("mapping.match_selection requires all, uniform or constraint");
        match_budget = this->declare_parameter<int>("mapping.match_budget", 600);
        rotation_length = this->declare_parameter<double>("mapping.rotation_length", 5.0);
        if (match_budget < 1 || !std::isfinite(rotation_length) || rotation_length <= 0)
            throw std::invalid_argument("mapping.match_budget and rotation_length must be positive");
        const auto csv = this->declare_parameter<string>("diagnostics.frame_csv", "");
        if (!csv.empty())
        {
            frame_metrics.open(csv);
            if (!frame_metrics) throw std::runtime_error("Cannot open frame diagnostics: " + csv);
            frame_metrics << "end_ns,status,input,representatives,candidates,selected,weakest_all,weakest_selected,"
                             "scan_patches,matched_patches,map_patches,map_ms,preprocess_ms,frame_ms,mean_abs_residual,x,y,z,"
                             "extrinsic_x,extrinsic_y,extrinsic_z,extrinsic_qx,extrinsic_qy,extrinsic_qz,"
                             "extrinsic_qw,normal_groups_before_after\n";
        }
        this->get_parameter_or<double>("filter_size_map",filter_size_map_min,0.5);
        if (plane_association) plane_map = fast_lio::LocalPlaneMap(filter_size_map_min, scan_options);
        this->get_parameter_or<double>("cube_side_length",cube_len,200.f);
        this->get_parameter_or<float>("mapping.det_range",DET_RANGE,300.f);
        this->get_parameter_or<double>("mapping.fov_degree",fov_deg,180.f);
        this->get_parameter_or<double>("mapping.gyr_cov",gyr_cov,0.1);
        this->get_parameter_or<double>("mapping.acc_cov",acc_cov,0.1);
        this->get_parameter_or<double>("mapping.b_gyr_cov",b_gyr_cov,0.0001);
        this->get_parameter_or<double>("mapping.b_acc_cov",b_acc_cov,0.0001);
        this->get_parameter_or<double>("preprocess.blind", p_pre->blind, 0.01);
        this->get_parameter_or<int>("preprocess.lidar_type", p_pre->lidar_type, AVIA);
        this->get_parameter_or<int>("preprocess.scan_line", p_pre->N_SCANS, 16);
        this->get_parameter_or<int>("preprocess.timestamp_unit", p_pre->time_unit, US);
        this->get_parameter_or<int>("preprocess.scan_rate", p_pre->SCAN_RATE, 10);
        this->get_parameter_or<int>("point_filter_num", p_pre->point_filter_num, 2);
        this->get_parameter_or<bool>("feature_extract_enable", p_pre->feature_enabled, false);
        this->get_parameter_or<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
        this->get_parameter_or<bool>("mapping.extrinsic_est_en", extrinsic_est_en, true);
        this->get_parameter_or<bool>("pcd_save.pcd_save_en", pcd_save_en, false);
        this->get_parameter_or<int>("pcd_save.interval", pcd_save_interval, -1);
        this->get_parameter_or<vector<double>>("mapping.extrinsic_T", extrinT, vector<double>());
        this->get_parameter_or<vector<double>>("mapping.extrinsic_R", extrinR, vector<double>());

        if (p_pre->point_filter_num < 1 || p_pre->N_SCANS < 1 || p_pre->N_SCANS > 128)
            throw std::invalid_argument("point_filter_num must be positive and preprocess.scan_line must be in [1,128]");
        RCLCPP_INFO(this->get_logger(), "p_pre->lidar_type %d", p_pre->lidar_type);

        path.header.stamp = this->get_clock()->now();
        path.header.frame_id ="camera_init";

        // /*** variables definition ***/
        // int effect_feat_num = 0, frame_num = 0;
        // double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
        // bool flg_EKF_converged, EKF_stop_flg = 0;

        FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
        HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

        _featsArray.reset(new PointCloudXYZI());

        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
        downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

        Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
        Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
        p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
        p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
        p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
        p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
        p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

        fill(epsi, epsi+23, 0.001);
        kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

        /*** debug record ***/
        // FILE *fp;
        string pos_log_dir = root_dir + "/Log/pos_log.txt";
        fp = fopen(pos_log_dir.c_str(),"w");

        // ofstream fout_pre, fout_out, fout_dbg;
        fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
        fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
        fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
        if (fout_pre && fout_out)
            cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
        else
            cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

        /*** ROS subscribe initialization ***/
        if (p_pre->lidar_type == AVIA)
        {
            sub_pcl_livox_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(lid_topic, 20, livox_pcl_cbk);
        }
        else
        {
            sub_pcl_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(lid_topic, rclcpp::SensorDataQoS(), standard_pcl_cbk);
        }
        timing_pub = create_publisher<FrameTiming>("lio/frame_timing", 200);
        if (declare_parameter<bool>("flight_output.enabled", false))
            flight_output_ = std::make_unique<fast_lio::FlightOdometry>(*this,
                [](int64_t end, int64_t enqueued, int64_t dequeued, int64_t published) {
                    if (enqueued) record_timing(end, FrameTiming::ENQUEUED, enqueued);
                    if (dequeued) record_timing(end, FrameTiming::DEQUEUED, dequeued);
                    if (published) record_timing(end, FrameTiming::PUBLISHED, published);
                });
        imu_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        rclcpp::SubscriptionOptions imu_options;
        imu_options.callback_group = imu_group_;
        sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(imu_topic, 100, buffer_imu, imu_options);
        pubLaserCloudFull_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 20);
        pubLaserCloudFull_body_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_body", 20);
        body_xyz_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_body_xyz", 3);
        pubLaserCloudEffect_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_effected", 20);
        pubLaserCloudMap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/Laser_map", 20);
        pubOdomAftMapped_ = this->create_publisher<nav_msgs::msg::Odometry>("/Odometry", 20);
        pubPath_ = this->create_publisher<nav_msgs::msg::Path>("/path", 20);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        //------------------------------------------------------------------------------------------------------
        auto period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0 / 100.0));
        timer_ = rclcpp::create_timer(this, this->get_clock(), period_ms, std::bind(&LaserMappingNode::timer_callback, this));

        auto map_period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0));
        map_pub_timer_ = rclcpp::create_timer(this, this->get_clock(), map_period_ms, std::bind(&LaserMappingNode::map_publish_callback, this));

        map_save_srv_ = this->create_service<std_srvs::srv::Trigger>("map_save", std::bind(&LaserMappingNode::map_save_callback, this, std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(this->get_logger(), "Node init finished.");
    }

    ~LaserMappingNode()
    {
        fout_out.close();
        fout_pre.close();
        fclose(fp);
    }

private:

    void timer_callback()
    {
        if(sync_packages(Measures))
        {
            record_timing(Measures.lidar_end_stamp.nanoseconds(), FrameTiming::SYNCHRONIZED, fast_lio::steady_ns());
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                return;
            }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();

            p_imu->Process(Measures, kf, feats_undistort);
            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
                return;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();

            /*** downsample the feature points in a scan ***/
            if (adaptive_frontend || plane_association || (scan_pub_en && adaptive_scan_pub_en && !dense_pub_en))
                fast_lio::buildAdaptiveScan(*feats_undistort, scan_options, *adaptive_representatives, adaptive_scan);
            if (adaptive_frontend || plane_association)
                feats_down_body = adaptive_representatives;
            else
            {
                downSizeFilterSurf.setInputCloud(feats_undistort);
                downSizeFilterSurf.filter(*feats_down_body);
            }
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();
            candidate_count = effct_feat_num = 0;
            matched_plane_count = 0;
            selection_stats = {};
            res_mean_last = 0;
            // Association verifies a combination of patches. Grow the initial seed
            // until the map can represent a combination, including partially sparse scans.
            if (plane_association && plane_map.size() < 2) {
                integrate_plane_scan();
                write_frame_metrics(plane_map.size() < 2 ? "bootstrap_wait" : "seed", t0, omp_get_wtime() - t1);
                return;
            }
            if (!plane_association && ikdtree.Root_Node == nullptr)
            {
                RCLCPP_INFO(this->get_logger(), "Initialize the map kdtree");
                if(feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for(int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    ikdtree.Build(feats_down_world->points);
                }
                write_frame_metrics(feats_down_size > 5 ? "seed" : "bootstrap_wait", t0, omp_get_wtime() - t1);
                return;
            }
            kdtree_size_st = plane_association ? plane_map.size() : ikdtree.size();
            
            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (!plane_association && feats_down_size < 5)
            {
                write_frame_metrics("insufficient", t0, 0.0);
                RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
                return;
            }
            
            if (!plane_association) {
                normvec->resize(feats_down_size);
                feats_down_world->resize(feats_down_size);
                point_selected_surf.assign(feats_down_size, 0);
                res_last.resize(feats_down_size);
                Nearest_Points.resize(feats_down_size);
            }

            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
            <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;

            if(0) // If you need to see map point, change to "if(1)"
            {
                PointVector ().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }


            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            const bool corrected = kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped_, tf_broadcaster_, corrected ? flight_output_.get() : nullptr);

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            if (plane_association) {
                if (corrected) integrate_plane_scan();
                kdtree_incremental_time = omp_get_wtime() - t3;
            } else {
                map_incremental();
            }
            t5 = omp_get_wtime();
            
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath_);
            if (scan_pub_en && corrected) publish_frame_world(pubLaserCloudFull_);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body_, body_xyz_pub_);
            if (effect_pub_en) publish_effect_world(pubLaserCloudEffect_);
            // if (map_pub_en) publish_map(pubLaserCloudMap_);

            write_frame_metrics(corrected ? "corrected" : "propagated", t0, t5 - t3);
            /*** Debug variables ***/
            if (runtime_pos_log && time_log_counter < MAXN)
            {
                frame_num ++;
                kdtree_size_end = plane_association ? plane_map.size() : ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                time_log_counter ++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
                dump_lio_state_to_log(fp);
            }
        }
    }

    void map_publish_callback()
    {
        if (map_pub_en) publish_map(pubLaserCloudMap_);
    }

    void map_save_callback(std_srvs::srv::Trigger::Request::ConstSharedPtr req, std_srvs::srv::Trigger::Response::SharedPtr res)
    {
        RCLCPP_INFO(this->get_logger(), "Saving map to %s...", map_file_path.c_str());
        if (pcd_save_en)
        {
            save_to_pcd();
            res->success = true;
            res->message = "Map saved.";
        }
        else
        {
            res->success = false;
            res->message = "Map save disabled.";
        }
    }

private:
    rclcpp::CallbackGroup::SharedPtr imu_group_;
    std::unique_ptr<fast_lio::FlightOdometry> flight_output_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr body_xyz_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_pc_;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr map_pub_timer_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr map_save_srv_;

    bool effect_pub_en = false, map_pub_en = false;
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    double epsi[23] = {0.001};

    FILE *fp;
    ofstream fout_pre, fout_out, fout_dbg;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    signal(SIGINT, SigHandle);

    {
        auto node = std::make_shared<LaserMappingNode>();
        rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
        executor.add_node(node);
        executor.spin();
    }

    if (rclcpp::ok())
        rclcpp::shutdown();
    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name<<endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;    
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(),"w");
        fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0;i<time_log_counter; i++){
            fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    return 0;
}

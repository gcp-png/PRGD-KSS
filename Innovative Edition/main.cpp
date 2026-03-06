#define _USE_MATH_DEFINES
#include "kss_icp.h"
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/registration/ia_ransac.h>
#include <iostream>
#include <string>
#include <cmath>
#include <thread>
#include <chrono>
#include <Eigen/Core>

// Point cloud validity check
bool checkCloudValid(const kss_icp::PointCloudT::Ptr& cloud) {
    if (!cloud) return false;
    for (const auto& p : cloud->points) {
        if (std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z) ||
            std::isinf(p.x) || std::isinf(p.y) || std::isinf(p.z)) {
            std::cerr << "Point cloud contains invalid values" << std::endl;
            return false;
        }
    }
    std::cout << "Point cloud data is valid, no invalid values" << std::endl;

    return true;
}

// Point cloud loading
bool loadPointCloud(const std::string& path, kss_icp::PointCloudT::Ptr& cloud) {
    if (!cloud) {
        cloud.reset(new kss_icp::PointCloudT);
    }

    cloud->clear();
    try {
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(path, *cloud) == -1) {
            std::cerr << "Cannot load point cloud file: " << path << std::endl;
            return false;
        }
        std::cout << "Successfully loaded point cloud: " << path << " (number of points: " << cloud->size() << ")" << std::endl;
        return checkCloudValid(cloud);
    }
    catch (const std::exception& e) {
        std::cerr << "Exception occurred while loading point cloud: " << e.what() << std::endl;
        return false;
    }
}

// Compute RMSE for point cloud registration (works with different point cloud sizes, suitable for industrial scenarios)
double computeRMSE(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudA, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudB)
{
    // Check if point clouds are empty
    if (cloudA->empty() || cloudB->empty())
    {
        std::cerr << "Error: One or both point clouds are empty!" << std::endl;
        return -1.0;
    }

    // Build KD-tree for target point cloud (accelerate nearest neighbor search)
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloudB);

    double sum_squared_dist = 0.0; // Store sum of squared distances for all points
    int valid_points = 0;          // Number of valid matches

    // Iterate through each point in source point cloud (cloudA) and find nearest neighbor in target point cloud (cloudB)
    for (size_t i = 0; i < cloudA->points.size(); ++i)
    {
        pcl::PointXYZ searchPoint = cloudA->points[i];
        std::vector<int> nearestIdx(1);    // Index of nearest neighbor
        std::vector<float> nearestDistSq(1);// Squared distance to nearest neighbor

        // Search for nearest neighbor (k=1 means find only 1 nearest point)
        int found = kdtree.nearestKSearch(searchPoint, 1, nearestIdx, nearestDistSq);

        // Ensure valid neighbor was found
        if (found > 0)
        {
            sum_squared_dist += nearestDistSq[0]; // Accumulate squared distance
            valid_points++;
        }
    }

    // Check valid point count (avoid division by zero)
    if (valid_points == 0)
    {
        std::cerr << "Error: No valid nearest neighbors found!" << std::endl;
        return -1.0;
    }

    // Compute MSE and RMSE
    double mse = sum_squared_dist / valid_points;
    double rmse = sqrt(mse);

    return rmse;
}

int main() {
    // Ensure Eigen uses proper memory alignment
    Eigen::initParallel();
#ifndef NDEBUG
    std::cout << "Debug mode: Memory alignment check enabled" << std::endl;
#endif
    std::string target_path = "C:/Users/86198/Desktop/PRGD-KSS/cup.pcd";
    std::string source_path = "C:/Users/86198/Desktop/PRGD-KSS/cup1.pcd";
    std::string output_path = "C:/Users/86198/Desktop/noise/rabbit11.pcd";
    kss_icp::PointCloudT::Ptr target(new kss_icp::PointCloudT);
    kss_icp::PointCloudT::Ptr source(new kss_icp::PointCloudT);

    if (!target || !source) {
        std::cerr << "Point cloud pointer initialization failed" << std::endl;
        return -1;
    }

    if (!loadPointCloud(target_path, target) || !loadPointCloud(source_path, source)) {
        std::cerr << "Point cloud loading failed" << std::endl;
        return -1;
    }
    clock_t start = clock();
    kss_icp::KSSICP kss_icp;
    kss_icp.setSampleSize(200);
    kss_icp.setVoxelScale(0.05f);
    kss_icp.setRotationStep(M_PI / 24);

    kss_icp::testKSSInvariance(kss_icp, source);

    Eigen::Matrix4f final_transform = Eigen::Matrix4f::Identity();
    kss_icp.align(target, source, final_transform);
    // clock_t start = clock();
    kss_icp::PointCloudT::Ptr aligned_source(new kss_icp::PointCloudT);
    pcl::transformPointCloud(*source, *aligned_source, final_transform);
    float final_dist = kss_icp.computeHausdorffDistance(target, aligned_source);
    clock_t end = clock();
    std::cout << "Total time: " << (double)(end - start) / (double)CLOCKS_PER_SEC << " s" << std::endl;
    std::cout << "=== Final Registration Error ===" << std::endl;
    std::cout << "Hausdorff distance between target cloud and aligned source cloud: " << final_dist << std::endl;

    double mse = computeRMSE(target, aligned_source);
    std::cout << "Root Mean Square Error (RMSE): " << mse << std::endl;
    if (final_dist < 0.05) {
        std::cout << "Final registration error acceptable" << std::endl;
    }
    else {
        std::cout << "Final registration error unacceptable" << std::endl;
    }

    if (pcl::io::savePCDFileBinary(output_path, *aligned_source) == -1) {
        std::cerr << "Failed to save registration result" << std::endl;
    }
    else {
        std::cout << "Registration result saved to: " << output_path << std::endl;
    }
    target.reset();
    source.reset();
    target.reset();
    source.reset();
    aligned_source.reset();

#ifdef _DEBUG
    std::cout << "Program execution completed, memory cleanup finished" << std::endl;
#endif
    std::cout << "About to return 0" << std::endl;
    std::cout.flush();
    return 0;
}
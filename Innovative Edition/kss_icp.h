#pragma once
#ifndef KSS_ICP_H
#define KSS_ICP_H
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/registration/icp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>
#include <pcl/common/centroid.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <memory>
namespace kss_icp {
    typedef pcl::PointXYZ PointT;
    typedef pcl::PointCloud<PointT> PointCloudT;
    typedef std::shared_ptr<PointCloudT> PointCloudPtr;
    typedef std::shared_ptr<const PointCloudT> PointCloudConstPtr;

    // Structure for multi-scale features
    struct MultiScaleFeature {
        Eigen::Matrix3f direction_matrix;  // Fused principal direction matrix
        float confidence;                  // Feature confidence
        std::vector<Eigen::Matrix3f> scale_directions; // Principal directions at each scale
        std::vector<float> scale_weights;  // Weights for each scale
    };

    class KSSICP {
    public:
        KSSICP();

        // Multi-scale feature fusion function
        MultiScaleFeature computeMultiScaleFeatures(
            const PointCloudPtr& pre_cloud,
            const std::vector<float>& scale_radii);

        Eigen::Matrix3f computeLocalDirection(
            const PointCloudPtr& cloud,
            const PointT& center_point,
            float radius,
            float& eigenvalue_ratio);

        Eigen::Matrix3f projectToSO3(const Eigen::Matrix3f& matrix);

        std::vector<Eigen::Matrix3f> generateMultiScaleInitialRotations(
            const PointCloudPtr& source_pre,
            const PointCloudPtr& target_pre);

        // Parameter setters
        void setSampleSize(int size) { sample_size_ = size; }
        void setVoxelScale(float scale) { voxel_scale_ = scale; }
        void setRotationStep(float step) { rotation_step_ = step; }

        bool isCloudValid(const PointCloudPtr& cloud);

        float getCloudScale1(const PointCloudT& cloud) {
            if (cloud.empty()) {
                return 0.0f;
            }
            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(cloud, centroid);
            float scale_sum = 0.0f;
            for (const auto& p : cloud) {
                scale_sum += (p.x - centroid[0]) * (p.x - centroid[0]) +
                    (p.y - centroid[1]) * (p.y - centroid[1]) +
                    (p.z - centroid[2]) * (p.z - centroid[2]);
            }
            // Prevent division by zero
            if (cloud.size() == 0) {
                return 0.0f;
            }
            return std::sqrt(scale_sum / static_cast<float>(cloud.size()));
        }

        // Perform registration
        void align(const PointCloudPtr& target,
            const PointCloudPtr& source,
            Eigen::Matrix4f& final_transform);

        void applyRotation(
            const PointCloudPtr& input,
            PointCloudPtr& output,
            const Eigen::Matrix3f& rotation
        );

        // Pre-shape space mapping and Hausdorff distance computation
        void mapToPreShapeSpace(const PointCloudPtr& input, PointCloudPtr& output,
            Eigen::Vector4f& centroid, float& scale);

        // Compute point cloud principal directions (PCA)
        Eigen::Matrix3f computePrincipalDirections(const PointCloudPtr& pre_cloud);

        // Generate initial rotation candidates based on PCA principal directions
        std::vector<Eigen::Matrix3f> generateInitialRotationsFromPCA(
            const PointCloudPtr& source_pre,
            const PointCloudPtr& target_pre);

        // Compute Euclidean gradient for RGD
        Eigen::Matrix3f computeEuclideanGradient(
            const Eigen::Matrix3f& R,
            const PointCloudPtr& source_pre,
            const PointCloudPtr& target_pre);

        // Riemannian gradient descent for rotation optimization
        Eigen::Matrix3f riemannianGradientDescent(
            const PointCloudPtr& source_pre,
            const PointCloudPtr& target_pre,
            const Eigen::Matrix3f& init_rot);

        float computeHausdorffDistance(const PointCloudPtr& cloud1,
            const PointCloudPtr& cloud2);

    private:
        // Point cloud downsampling
        void downsampleCloud(const PointCloudPtr& input, PointCloudPtr& output);

        // Generate rotation candidates
        std::vector<Eigen::Matrix3f> generateRotationCandidates();

        // Parameters
        std::vector<float> scale_radii_ = { 0.05f, 0.1f, 0.2f }; // Multi-scale radii
        float min_eigenvalue_ratio_ = 2.0f; // Eigenvalue ratio threshold

        // Parameters
        int sample_size_;       // Number of sample points
        float voxel_scale_;     // Voxel scale
        float rotation_step_;   // Rotation step size (radians)
    };
    // Test function
    void testKSSInvariance(KSSICP& kss_icp, const PointCloudPtr& source_raw);
}

#endif // KSS_ICP_H
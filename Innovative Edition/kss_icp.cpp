#include "kss_icp.h"
#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>
#include <omp.h>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <limits>
#include <algorithm>
#include <cassert>

namespace kss_icp {
    //computeLocalDirection
    KSSICP::KSSICP()
        : sample_size_(300), voxel_scale_(1.0f), rotation_step_(M_PI / 36), min_eigenvalue_ratio_(2.0f) {
        // Set multi-scale radii
        scale_radii_ = { 0.05f, 0.1f, 0.2f, 0.3f };
        Eigen::initParallel();

    }
    // Point cloud validity check
    bool KSSICP::isCloudValid(const PointCloudPtr& cloud) {
        if (!cloud || cloud->empty()) return false;
        for (const auto& p : *cloud) {
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
                return false;
            }
        }
        return true;
    }

    // Point cloud downsampling
    void KSSICP::downsampleCloud(const PointCloudPtr& input, PointCloudPtr& output) {
        if (!input || input->empty()) {
            output.reset(new PointCloudT);
            return;
        }

        if (!output) {
            output.reset(new PointCloudT);
        }

        PointCloudPtr voxel_filtered(new PointCloudT);
        pcl::VoxelGrid<PointT> voxel;
        voxel.setInputCloud(input);
        voxel.setLeafSize(voxel_scale_, voxel_scale_, voxel_scale_);
        voxel.filter(*voxel_filtered);

        if (voxel_filtered->size() > static_cast<size_t>(sample_size_)) {
            pcl::UniformSampling<PointT> uniform;
            uniform.setInputCloud(voxel_filtered);
            uniform.setRadiusSearch(0.05);

            PointCloudPtr uniform_filtered(new PointCloudT);
            uniform.filter(*uniform_filtered);

            if (uniform_filtered->size() > static_cast<size_t>(sample_size_)) {
                output->clear();
                output->reserve(sample_size_);
                size_t copy_size = std::min(static_cast<size_t>(sample_size_),
                    uniform_filtered->size());
                for (size_t i = 0; i < copy_size; ++i) {
                    output->push_back((*uniform_filtered)[i]);
                }
            }
            else {
                *output = *uniform_filtered;
            }
        }
        else {
            *output = *voxel_filtered;
        }

        std::cout << "Number of points after simplification: " << output->size() << std::endl;
    }
    // Compute principal direction of local neighborhood
    Eigen::Matrix3f KSSICP::computeLocalDirection(
        const PointCloudPtr& cloud,
        const PointT& center_point,
        float radius,
        float& eigenvalue_ratio) {

        if (!cloud || cloud->empty()) {
            eigenvalue_ratio = 0.0f;
            return Eigen::Matrix3f::Identity();
        }

        // 1. Extract points within radius neighborhood
        std::vector<int> point_indices;
        std::vector<float> point_distances;

        pcl::KdTreeFLANN<PointT> kdtree;
        kdtree.setInputCloud(cloud);

        if (kdtree.radiusSearch(center_point, radius, point_indices, point_distances) < 3) {
            eigenvalue_ratio = 0.0f;
            return Eigen::Matrix3f::Identity();
        }

        // 2. Construct local point cloud
        PointCloudPtr local_cloud(new PointCloudT);
        for (int idx : point_indices) {
            local_cloud->push_back((*cloud)[idx]);
        }

        // 3. Compute weighted covariance matrix
        Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();

        // Compute centroid
        for (const auto& p : *local_cloud) {
            centroid += Eigen::Vector3f(p.x, p.y, p.z);
        }
        centroid /= static_cast<float>(local_cloud->size());

        // Compute covariance matrix (using Gaussian weights)
        float sigma = radius / 3.0f;
        for (const auto& p : *local_cloud) {
            Eigen::Vector3f vec(p.x - centroid.x(),
                p.y - centroid.y(),
                p.z - centroid.z());
            float dist_sq = vec.squaredNorm();
            float weight = std::exp(-dist_sq / (2 * sigma * sigma));
            covariance += weight * vec * vec.transpose();
        }

        // 4. Eigenvalue decomposition
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance);
        Eigen::Vector3f eigenvalues = eigen_solver.eigenvalues();
        Eigen::Matrix3f eigenvectors = eigen_solver.eigenvectors();

        // Sort in descending order of eigenvalues
        std::vector<int> idx_sorted = { 0, 1, 2 };
        std::sort(idx_sorted.begin(), idx_sorted.end(),
            [&eigenvalues](int i, int j) { return eigenvalues[i] > eigenvalues[j]; });

        // Compute eigenvalue ratio
        eigenvalue_ratio = (eigenvalues[idx_sorted[1]] > 1e-10f) ?
            eigenvalues[idx_sorted[0]] / eigenvalues[idx_sorted[1]] : 0.0f;

        // 5. Construct direction matrix
        Eigen::Matrix3f direction_matrix;
        for (int i = 0; i < 3; ++i) {
            direction_matrix.col(i) = eigenvectors.col(idx_sorted[i]);
        }

        // Ensure right-handed coordinate system
        if (direction_matrix.determinant() < 0) {
            direction_matrix.col(2) *= -1;
        }

        return direction_matrix;
    }

    // Project to SO(3)
    Eigen::Matrix3f KSSICP::projectToSO3(const Eigen::Matrix3f& matrix) {
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(matrix,
            Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3f R = svd.matrixU() * svd.matrixV().transpose();
        if (R.determinant() < 0) {
            R.col(2) *= -1;
        }
        return R;
    }
    // Compute multi-scale fusion features
    MultiScaleFeature KSSICP::computeMultiScaleFeatures(
        const PointCloudPtr& pre_cloud,
        const std::vector<float>& scale_radii) {

        MultiScaleFeature feature;

        if (!pre_cloud || pre_cloud->empty() || scale_radii.empty()) {
            feature.direction_matrix = Eigen::Matrix3f::Identity();
            feature.confidence = 0.0f;
            return feature;
        }

        // 1. Compute multi-scale features for each point
        std::vector<Eigen::Matrix3f> all_directions;
        std::vector<float> all_weights;

        for (const auto& point : *pre_cloud) {
            // Compute principal directions at each scale for this point
            for (size_t i = 0; i < scale_radii.size(); ++i) {
                float radius = scale_radii[i];
                float eigenvalue_ratio;

                Eigen::Matrix3f dir = computeLocalDirection(pre_cloud, point, radius, eigenvalue_ratio);

                if (eigenvalue_ratio > min_eigenvalue_ratio_) {
                    feature.scale_directions.push_back(dir);
                    feature.scale_weights.push_back(eigenvalue_ratio);
                    all_directions.push_back(dir);
                    all_weights.push_back(eigenvalue_ratio);
                }
            }
        }

        if (all_directions.empty()) {
            feature.direction_matrix = Eigen::Matrix3f::Identity();
            feature.confidence = 0.0f;
            return feature;
        }

        // 2. Weighted average fusion
        Eigen::Matrix3f fused_matrix = Eigen::Matrix3f::Zero();
        float total_weight = 0.0f;

        for (size_t i = 0; i < all_directions.size(); ++i) {
            fused_matrix += all_weights[i] * all_directions[i];
            total_weight += all_weights[i];
        }

        if (total_weight > 1e-10f) {
            fused_matrix /= total_weight;
        }

        // 3. Project to SO(3)
        feature.direction_matrix = projectToSO3(fused_matrix);
        feature.confidence = total_weight / (all_directions.size() + 1e-10f);

        return feature;
    }

    // Pre-shape space mapping
    void KSSICP::mapToPreShapeSpace(const PointCloudPtr& input, PointCloudPtr& output,
        Eigen::Vector4f& centroid, float& scale) {
        if (!input || input->empty()) {
            output.reset(new PointCloudT);
            return;
        }

        if (!output) {
            output.reset(new PointCloudT);
        }

        pcl::compute3DCentroid(*input, centroid);

        PointCloudPtr centered(new PointCloudT);
        centered->reserve(input->size());
        for (const auto& p : *input) {
            PointT pt;
            pt.x = p.x - centroid[0];
            pt.y = p.y - centroid[1];
            pt.z = p.z - centroid[2];
            centered->push_back(pt);
        }

        scale = 0.0f;
        for (const auto& p : *centered) {
            scale += p.x * p.x + p.y * p.y + p.z * p.z;
        }
        scale = std::sqrt(scale);

        if (scale < 1e-10f) {
            scale = 1.0f;
        }

        output->clear();
        output->reserve(centered->size());
        for (const auto& p : *centered) {
            PointT pt;
            pt.x = p.x / scale;
            pt.y = p.y / scale;
            pt.z = p.z / scale;
            output->push_back(pt);
        }

        // KSS verification
        std::cout << "\n=== Pre-shape Space Mapping Verification ===" << std::endl;
        Eigen::Vector4f pre_centroid;
        pcl::compute3DCentroid(*output, pre_centroid);
        std::cout << "1. Pre-shape point cloud centroid: (" << pre_centroid[0] << ", "
            << pre_centroid[1] << ", " << pre_centroid[2] << ")" << std::endl;
        float energy = 0.0f;
        for (const auto& p : *output) {
            energy += p.x * p.x + p.y * p.y + p.z * p.z;
        }
        std::cout << "2. Pre-shape point cloud energy (sum(||p||²)): " << energy << std::endl;
        std::cout << "===============================================\n" << std::endl;
    }
    // Generate rotation candidates
    std::vector<Eigen::Matrix3f> KSSICP::generateRotationCandidates() {
        std::vector<Eigen::Matrix3f> rotations;
        float step = rotation_step_;

        for (float roll = -M_PI; roll <= M_PI; roll += step) {
            for (float pitch = -M_PI / 2; pitch <= M_PI / 2; pitch += step) {
                for (float yaw = -M_PI; yaw <= M_PI; yaw += step) {
                    Eigen::Quaternionf q = Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX())
                        * Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY())
                        * Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ());
                    rotations.push_back(q.toRotationMatrix());
                }
            }
        }

        return rotations;
    }
    // Compute Hausdorff distance
    float KSSICP::computeHausdorffDistance(const PointCloudPtr& cloud1,
        const PointCloudPtr& cloud2) {
        if (!cloud1 || !cloud2 || cloud1->empty() || cloud2->empty()) {
            return 0.0f;
        }

        pcl::KdTreeFLANN<PointT> kdtree;
        kdtree.setInputCloud(cloud2);

        float max_dist1 = 0.0f;
        int k = 1;
        std::vector<int> idx(k);
        std::vector<float> dist_sq(k);

        for (const auto& p : *cloud1) {
            if (kdtree.nearestKSearch(p, k, idx, dist_sq) > 0) {
                float dist = std::sqrt(dist_sq[0]);
                if (dist > max_dist1) max_dist1 = dist;
            }
        }
        kdtree.setInputCloud(cloud1);
        float max_dist2 = 0.0f;
        for (const auto& p : *cloud2) {
            if (kdtree.nearestKSearch(p, k, idx, dist_sq) > 0) {
                float dist = std::sqrt(dist_sq[0]);
                if (dist > max_dist2) max_dist2 = dist;
            }
        }

        return std::max(max_dist1, max_dist2);
    }
    // Apply rotation
    void KSSICP::applyRotation(
        const PointCloudPtr& input,
        PointCloudPtr& output,
        const Eigen::Matrix3f& rotation
    ) {
        if (!input || input->empty()) {
            output.reset(new PointCloudT);
            return;
        }

        if (!output) {
            output.reset(new PointCloudT);
        }

        output->clear();
        output->reserve(input->size());

        for (const auto& p : *input) {
            Eigen::Vector3f point(p.x, p.y, p.z);
            Eigen::Vector3f rotated_point = rotation * point;

            PointT pt;
            pt.x = rotated_point.x();
            pt.y = rotated_point.y();
            pt.z = rotated_point.z();
            output->push_back(pt);
        }
    }

    // Compute principal directions of KSS pre-shape point cloud (PCA)
    Eigen::Matrix3f KSSICP::computePrincipalDirections(const PointCloudPtr& pre_cloud) {
        if (!pre_cloud || pre_cloud->empty()) {
            return Eigen::Matrix3f::Identity();
        }
        // Apply statistical filtering to pre-shape point cloud
        PointCloudPtr cloud_filtered(new PointCloudT);
        pcl::StatisticalOutlierRemoval<PointT> sor;
        sor.setInputCloud(pre_cloud);
        sor.setMeanK(20);
        sor.setStddevMulThresh(1.0);
        sor.filter(*cloud_filtered);

        if (cloud_filtered->empty()) {
            cloud_filtered = pre_cloud;
        }

        // Compute PCA based on filtered point cloud
        Eigen::Matrix3f covariance;
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cloud_filtered, centroid);
        pcl::computeCovarianceMatrixNormalized(*cloud_filtered, centroid, covariance);

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
        return eigen_solver.eigenvectors().leftCols(3).eval();
    }

    // Generate initial rotation candidates based on PCA principal directions
    std::vector<Eigen::Matrix3f> KSSICP::generateInitialRotationsFromPCA(
        const PointCloudPtr& source_pre,
        const PointCloudPtr& target_pre) {
        std::vector<Eigen::Matrix3f> initial_rots;

        if (!source_pre || !target_pre || source_pre->empty() || target_pre->empty()) {
            initial_rots.push_back(Eigen::Matrix3f::Identity());
            return initial_rots;
        }

        Eigen::Matrix3f S = computePrincipalDirections(source_pre);
        Eigen::Matrix3f T = computePrincipalDirections(target_pre);

        // Generate 6 candidate rotations
        initial_rots.push_back(T * S.transpose());

        Eigen::Matrix3f T_flipX = T;
        T_flipX.col(0) *= -1;
        initial_rots.push_back(T_flipX * S.transpose());

        Eigen::Matrix3f T_flipY = T;
        T_flipY.col(1) *= -1;
        initial_rots.push_back(T_flipY * S.transpose());

        Eigen::Matrix3f T_flipZ = T;
        T_flipZ.col(2) *= -1;
        initial_rots.push_back(T_flipZ * S.transpose());

        Eigen::Matrix3f T_flipXY = T;
        T_flipXY.col(0) *= -1;
        T_flipXY.col(1) *= -1;
        initial_rots.push_back(T_flipXY * S.transpose());

        Eigen::Matrix3f T_flipYZ = T;
        T_flipYZ.col(1) *= -1;
        T_flipYZ.col(2) *= -1;
        initial_rots.push_back(T_flipYZ * S.transpose());

        // Orthogonalization correction
        for (auto& rot : initial_rots) {
            Eigen::JacobiSVD<Eigen::Matrix3f> svd(rot, Eigen::ComputeFullU | Eigen::ComputeFullV);
            rot = svd.matrixU() * svd.matrixV().transpose();
            if (rot.determinant() < 0) rot.col(2) *= -1;
        }

        return initial_rots;
    }
    // Replace the original generateInitialRotationsFromPCA function
    std::vector<Eigen::Matrix3f> KSSICP::generateMultiScaleInitialRotations(
        const PointCloudPtr& source_pre,
        const PointCloudPtr& target_pre) {

        std::vector<Eigen::Matrix3f> initial_rots;

        if (!source_pre || !target_pre || source_pre->empty() || target_pre->empty()) {
            initial_rots.push_back(Eigen::Matrix3f::Identity());
            return initial_rots;
        }

        // 1. Compute multi-scale fusion features
        MultiScaleFeature source_feature = computeMultiScaleFeatures(source_pre, scale_radii_);
        MultiScaleFeature target_feature = computeMultiScaleFeatures(target_pre, scale_radii_);

        std::cout << "Multi-scale feature confidence - Source: " << source_feature.confidence
            << ", Target: " << target_feature.confidence << std::endl;

        // 2. Generate rotation candidates (based on fusion features)
        if (source_feature.confidence > 0.3f && target_feature.confidence > 0.3f) {
            // Use multi-scale fusion features
            Eigen::Matrix3f S = source_feature.direction_matrix;
            Eigen::Matrix3f T = target_feature.direction_matrix;

            initial_rots.push_back(projectToSO3(T * S.transpose()));

            // Handle sign ambiguity
            Eigen::Matrix3f T_flipX = T;
            T_flipX.col(0) *= -1;
            initial_rots.push_back(projectToSO3(T_flipX * S.transpose()));

            Eigen::Matrix3f T_flipY = T;
            T_flipY.col(1) *= -1;
            initial_rots.push_back(projectToSO3(T_flipY * S.transpose()));

            Eigen::Matrix3f T_flipZ = T;
            T_flipZ.col(2) *= -1;
            initial_rots.push_back(projectToSO3(T_flipZ * S.transpose()));
        }
        else {
            // Fallback to traditional PCA
            std::cout << "Multi-scale feature confidence insufficient, falling back to PCA" << std::endl;
            initial_rots = generateInitialRotationsFromPCA(source_pre, target_pre);
        }

        // 3. Add random perturbation candidates
        for (size_t i = 0; i < std::min(initial_rots.size(), size_t(3)); ++i) {
            Eigen::AngleAxisf perturb(5.0f * M_PI / 180.0f,
                Eigen::Vector3f::Random().normalized());
            initial_rots.push_back(projectToSO3(perturb.toRotationMatrix() * initial_rots[i]));
        }

        return initial_rots;
    }
    // Compute Euclidean gradient for RGD
    Eigen::Matrix3f KSSICP::computeEuclideanGradient(
        const Eigen::Matrix3f& R,
        const PointCloudPtr& source_pre,
        const PointCloudPtr& target_pre) {
        Eigen::Matrix3f grad = Eigen::Matrix3f::Zero();

        if (!source_pre || !target_pre || source_pre->empty() || target_pre->empty()) {
            return grad;
        }
        // Use the smaller point cloud size to prevent out-of-bounds
        size_t N = std::min(source_pre->size(), target_pre->size());
        if (N == 0) {
            return grad;
        }
        for (size_t i = 0; i < N; ++i) {
            const auto& s = (*source_pre)[i];
            const auto& t = (*target_pre)[i];

            Eigen::Vector3f s_vec(s.x, s.y, s.z);
            Eigen::Vector3f Rs = R * s_vec;
            Eigen::Vector3f t_vec(t.x, t.y, t.z);

            grad += (Rs - t_vec) * s_vec.transpose();
        }

        return 2.0f / static_cast<float>(N) * grad;
    }
    // Riemannian Gradient Descent (RGD) for rotation matrix optimization
    Eigen::Matrix3f KSSICP::riemannianGradientDescent(
        const PointCloudPtr& source_pre,
        const PointCloudPtr& target_pre,
        const Eigen::Matrix3f& init_rot) {

        Eigen::Matrix3f R = init_rot;

        if (!source_pre || !target_pre || source_pre->empty() || target_pre->empty()) {
            return R;
        }
        int max_iter = 100;
        float lr = 0.15f;
        float momentum = 0.9f;
        Eigen::Matrix3f prev_update = Eigen::Matrix3f::Identity();
        float best_loss = std::numeric_limits<float>::max();
        Eigen::Matrix3f best_R = R;

        for (int iter = 0; iter < max_iter; ++iter) {
            PointCloudPtr rotated_source(new PointCloudT);
            applyRotation(source_pre, rotated_source, R);
            float curr_loss = computeHausdorffDistance(rotated_source, target_pre);

            // Record best state
            if (curr_loss < best_loss) {
                best_loss = curr_loss;
                best_R = R;
            }
            // Convergence check
            if (iter > 0 && std::abs(curr_loss - best_loss) < 1e-9) {
                break;
            }

            // Compute gradient
            Eigen::Matrix3f euclid_grad = computeEuclideanGradient(R, source_pre, target_pre);
            Eigen::Matrix3f riemann_grad = 0.5f * (euclid_grad * R.transpose() - R * euclid_grad.transpose());

            // Exponential map update
            Eigen::AngleAxisf grad_angle_axis(riemann_grad);
            Eigen::Matrix3f curr_update = Eigen::AngleAxisf(-lr * grad_angle_axis.angle(),
                grad_angle_axis.axis()).toRotationMatrix();
            Eigen::Matrix3f update = prev_update * curr_update;
            R = R * update;
            prev_update = update;

            // Learning rate decay
            static int bad_iter = 0;
            if (curr_loss > best_loss) {
                bad_iter++;
                if (bad_iter >= 3) {
                    lr *= 0.3f;
                    bad_iter = 0;
                }
            }
            else {
                bad_iter = 0;
            }
        }

        return best_R;
    }
    // Comparison visualization
    void visualizeCompare(const PointCloudT::Ptr& cloud1, const std::string& name1,
        const PointCloudT::Ptr& cloud2, const std::string& name2,
        const std::string& title, int duration = 3000) {
        try {
            pcl::visualization::PCLVisualizer viewer(title);
            viewer.setBackgroundColor(1, 1, 1);
            std::string id1 = name1 + "_pc";
            std::string id2 = name2 + "_pc";
            pcl::visualization::PointCloudColorHandlerCustom<PointT> color1(cloud1, 0, 0, 255);
            pcl::visualization::PointCloudColorHandlerCustom<PointT> color2(cloud2, 255, 0, 0);
            viewer.addPointCloud(cloud1, color1, id1);
            viewer.addPointCloud(cloud2, color2, id2);
            viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, id1);
            viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, id2);
            viewer.addCoordinateSystem(0.1);
            //viewer.addText(name1 + " (Red) vs " + name2 + " (Green)", 10, 10, 12, 1.0f, 1.0f, 1.0f, "legend");
            viewer.spinOnce(100000); // Display for 10 seconds
        }
        catch (const std::exception& e) {
            std::cerr << title << " comparison visualization failed: " << e.what() << std::endl;
        }
    }
    // Core align function
    void KSSICP::align(const PointCloudPtr& target,
        const PointCloudPtr& source,
        Eigen::Matrix4f& final_transform) {

        // Input validation
        if (!isCloudValid(target) || !isCloudValid(source)) {
            final_transform = Eigen::Matrix4f::Identity();
            std::cerr << "=== Alignment failed: Point cloud contains invalid values (NaN/Inf) ===" << std::endl;
            return;
        }

        if (!target || !source || target->empty() || source->empty()) {
            final_transform = Eigen::Matrix4f::Identity();
            std::cerr << "=== Alignment failed: Target/Source point cloud is empty ===" << std::endl;
            return;
        }

        std::cout << "=== Point cloud validity check passed ===" << std::endl;
        std::cout << "Target point cloud size: " << target->size() << std::endl;
        std::cout << "Source point cloud size: " << source->size() << std::endl;

        // Step 1: Point cloud downsampling
        PointCloudPtr target_simplified(new PointCloudT);
        PointCloudPtr source_simplified(new PointCloudT);
        downsampleCloud(target, target_simplified);
        downsampleCloud(source, source_simplified);

        std::cout << "\n=== Point Cloud Downsampling Results ===" << std::endl;
        std::cout << "Target point cloud size after simplification: " << target_simplified->size() << std::endl;
        std::cout << "Source point cloud size after simplification: " << source_simplified->size() << std::endl;
        std::cout << "===========================================\n" << std::endl;
        /*   visualizeCompare(source_simplified, "source",
               target_simplified, "target",
               "ICP Input Point Cloud Comparison");*/

               // Simplified cloud parameters
        PointCloudPtr target_pre(new PointCloudT);
        PointCloudPtr source_pre(new PointCloudT);
        Eigen::Vector4f target_simp_centroid, source_simp_centroid;
        float target_simp_scale, source_simp_scale;
        mapToPreShapeSpace(target_simplified, target_pre, target_simp_centroid, target_simp_scale);
        mapToPreShapeSpace(source_simplified, source_pre, source_simp_centroid, source_simp_scale);

        // Original point cloud parameter calculation
        std::cout << "\n=== Original Point Cloud Parameter Calculation ===" << std::endl;
        Eigen::Vector4f target_raw_centroid, source_raw_centroid;
        float target_raw_scale, source_raw_scale;

        // Compute original target point cloud centroid
        pcl::compute3DCentroid(*target, target_raw_centroid);

        // Compute original target point cloud scale
        target_raw_scale = 0.0f;
        size_t target_size = target->size();
        for (const auto& p : *target) {
            float dx = p.x - target_raw_centroid[0];
            float dy = p.y - target_raw_centroid[1];
            float dz = p.z - target_raw_centroid[2];
            target_raw_scale += dx * dx + dy * dy + dz * dz;
        }
        // Prevent division by zero
        if (target_size > 0) {
            target_raw_scale = std::sqrt(target_raw_scale / static_cast<float>(target_size));
        }
        else {
            target_raw_scale = 1.0f;
        }

        // Compute original source point cloud centroid and scale
        pcl::compute3DCentroid(*source, source_raw_centroid);
        source_raw_scale = 0.0f;
        size_t source_size = source->size();
        for (const auto& p : *source) {
            float dx = p.x - source_raw_centroid[0];
            float dy = p.y - source_raw_centroid[1];
            float dz = p.z - source_raw_centroid[2];
            source_raw_scale += dx * dx + dy * dy + dz * dz;
        }
        // Prevent division by zero
        if (source_size > 0) {
            source_raw_scale = std::sqrt(source_raw_scale / static_cast<float>(source_size));
        }
        else {
            source_raw_scale = 1.0f;
        }

        // Scale recovery ratio for original point clouds
        float scale_ratio = 1.0f;
        if (source_raw_scale > 1e-10f) {
            scale_ratio = target_raw_scale / source_raw_scale;
        }

        std::cout << "Original target point cloud centroid: (" << target_raw_centroid[0] << ","
            << target_raw_centroid[1] << "," << target_raw_centroid[2] << ")" << std::endl;
        std::cout << "Original source point cloud centroid: (" << source_raw_centroid[0] << ","
            << source_raw_centroid[1] << "," << source_raw_centroid[2] << ")" << std::endl;
        std::cout << "Original target point cloud scale: " << target_raw_scale << std::endl;
        std::cout << "Original source point cloud scale: " << source_raw_scale << std::endl;
        std::cout << "Original point cloud scale recovery ratio: " << scale_ratio << std::endl;
        std::cout << "=================================================\n" << std::endl;

        // Generate rotation candidates and select optimal rotation
        //std::vector<Eigen::Matrix3f> initial_rots = generateInitialRotationsFromPCA(source_pre, target_pre);
        std::vector<Eigen::Matrix3f> initial_rots = generateMultiScaleInitialRotations(source_pre, target_pre);
        std::cout << "Number of generated rotation candidates: " << initial_rots.size() << std::endl;
        float min_distance = std::numeric_limits<float>::max();
        Eigen::Matrix3f best_rotation = Eigen::Matrix3f::Identity();

        for (const auto& init_rot : initial_rots) {
            // First RGD optimization
            Eigen::Matrix3f optimized_rot1 = riemannianGradientDescent(source_pre, target_pre, init_rot);

            // Apply small perturbation to first result, perform second RGD
            Eigen::AngleAxisf perturb(1.0f * M_PI / 180.0f, Eigen::Vector3f::Random().normalized());
            Eigen::Matrix3f perturbed_rot = perturb.toRotationMatrix() * optimized_rot1;
            Eigen::Matrix3f optimized_rot2 = riemannianGradientDescent(source_pre, target_pre, perturbed_rot);

            // Evaluate both results
            PointCloudPtr rotated1(new PointCloudT);
            applyRotation(source_pre, rotated1, optimized_rot1);
            float dist1 = computeHausdorffDistance(target_pre, rotated1);

            PointCloudPtr rotated2(new PointCloudT);
            applyRotation(source_pre, rotated2, optimized_rot2);
            float dist2 = computeHausdorffDistance(target_pre, rotated2);

            // Update best rotation
            if (dist1 < min_distance) {
                min_distance = dist1;
                best_rotation = optimized_rot1;
            }
            if (dist2 < min_distance) {
                min_distance = dist2;
                best_rotation = optimized_rot2;
            }
        }

        std::cout << "=== PCA+RGD Optimal Rotation Search Completed ===" << std::endl;
        std::cout << "Optimal initial rotation distance: " << min_distance << std::endl;

        // Verify optimal rotation effect
        PointCloudPtr source_pre_rotated(new PointCloudT);
        applyRotation(source_pre, source_pre_rotated, best_rotation);
        float pre_align_dist = computeHausdorffDistance(target_pre, source_pre_rotated);

        std::cout << "\n=== KSS Alignment Effect Verification ===" << std::endl;
        std::cout << "1. Pre-shape point cloud distance after optimal rotation: " << pre_align_dist << std::endl;

        // KSS scale recovery logic
        // 1. Remove centroid from source point cloud
        PointCloudPtr source_centered(new PointCloudT);
        source_centered->reserve(source->size());
        for (const auto& p : *source) {
            PointT pt;
            pt.x = p.x - source_raw_centroid[0];
            pt.y = p.y - source_raw_centroid[1];
            pt.z = p.z - source_raw_centroid[2];
            source_centered->push_back(pt);
        }

        // 2. Scale by ratio
        PointCloudPtr source_scaled(new PointCloudT);
        source_scaled->reserve(source_centered->size());
        for (const auto& p : *source_centered) {
            PointT pt;
            pt.x = p.x * scale_ratio;
            pt.y = p.y * scale_ratio;
            pt.z = p.z * scale_ratio;
            source_scaled->push_back(pt);
        }

        // 3. Apply optimal rotation
        PointCloudPtr source_rotated(new PointCloudT);
        applyRotation(source_scaled, source_rotated, best_rotation);

        // 4. Add target centroid
        PointCloudPtr source_kss_aligned(new PointCloudT);
        source_kss_aligned->reserve(source_rotated->size());
        for (const auto& p : *source_rotated) {
            PointT pt;
            pt.x = p.x + target_raw_centroid[0];
            pt.y = p.y + target_raw_centroid[1];
            pt.z = p.z + target_raw_centroid[2];
            source_kss_aligned->push_back(pt);
        }
        // KSS to ICP transition verification
        std::cout << "\n=== KSS to ICP Transition Verification: Point Cloud Scale ===" << std::endl;
        float kss_aligned_scale = getCloudScale1(*source_kss_aligned);
        float target_scale = getCloudScale1(*target);
        std::cout << "Source point cloud scale after KSS processing: " << kss_aligned_scale << std::endl;
        std::cout << "Target point cloud scale: " << target_scale << std::endl;
        // T_kss matrix construction
        std::cout << "\n=== New Verification 1: T_kss (KSS coarse registration transform matrix) Validity ===" << std::endl;
        Eigen::Matrix4f T_kss = Eigen::Matrix4f::Identity();
        Eigen::Matrix3f R_scaled = scale_ratio * best_rotation;
        T_kss.block<3, 3>(0, 0) = R_scaled;
        Eigen::Vector3f trans_vec = target_raw_centroid.head<3>() - R_scaled * source_raw_centroid.head<3>();
        T_kss.block<3, 1>(0, 3) = trans_vec;
        /*visualizeCompare(target, "target",
            source_kss_aligned, "icp_source",
            "ICP Input Point Cloud Comparison");*/
        PointCloudPtr source_icp(new PointCloudT);
        downsampleCloud(source_kss_aligned, source_icp);
        PointCloudPtr target_icp(new PointCloudT);
        downsampleCloud(target, target_icp);
        // ICP fine registration
        std::cout << "\n=== New Verification 2: ICP Initial Transform Validation ===" << std::endl;
        pcl::IterativeClosestPoint<PointT, PointT> icp;
        icp.setInputSource(source_kss_aligned);
        icp.setInputTarget(target);
        icp.setMaximumIterations(150);
        icp.setTransformationEpsilon(1e-9);
        float max_corr_dist = target_scale * 0.03f;
        // if (max_corr_dist > 10.0f) max_corr_dist = 10.0f;  // Limit maximum distance
        icp.setMaxCorrespondenceDistance(10.0);
        icp.setRANSACOutlierRejectionThreshold(1.0);
        icp.setRANSACIterations(100);
        Eigen::Matrix4f icp_init_transform = Eigen::Matrix4f::Identity();
        std::cout << "ICP maximum correspondence distance threshold: " << max_corr_dist << std::endl;
        PointCloudT::Ptr aligned(new PointCloudT);
        icp.align(*aligned, icp_init_transform);
        std::cout << "ICP initial transform transfer completed" << std::endl;

        // Process ICP results
        std::cout << "\n=== ICP Fine Registration Results ===" << std::endl;
        std::cout << "ICP converged: " << (icp.hasConverged() ? "Yes" : "No") << std::endl;

        Eigen::Matrix4f full_final_transform = T_kss;

        if (icp.hasConverged()) {
            std::cout << "ICP Fitness Score (average correspondence distance): " << icp.getFitnessScore() << std::endl;
            Eigen::Matrix4f icp_transform = icp.getFinalTransformation();
            Eigen::Matrix3f rot_mat = icp_transform.block<3, 3>(0, 0);

            // Force orthogonalization of rotation matrix
            Eigen::JacobiSVD<Eigen::Matrix3f> svd(rot_mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3f rot_mat_ortho = svd.matrixU() * svd.matrixV().transpose();
            if (rot_mat_ortho.determinant() < 0) {
                rot_mat_ortho.col(2) *= -1;
            }

            icp_transform.block<3, 3>(0, 0) = rot_mat_ortho;
            final_transform = icp_transform;
            full_final_transform = icp_transform * T_kss;
        }
        else {
            std::cerr << "ICP did not converge, using KSS coarse alignment result" << std::endl;
            final_transform = Eigen::Matrix4f::Identity();
            full_final_transform = T_kss;
        }

        // Final transform
        final_transform = full_final_transform;
        // Visualize final result (original point clouds)
        PointCloudPtr source_final_aligned(new PointCloudT);
        pcl::transformPointCloud(*source, *source_final_aligned, full_final_transform);
        // Visualize ICP final result
        visualizeCompare(target, "target",
            source_final_aligned, "aligned_source",
            "ICP Fine Registration Final Result", 100000);
        std::cout << "\nFinal transformation matrix: " << std::endl << full_final_transform << std::endl;
        std::cout << "=========================================\n" << std::endl;
    }

    // KSS invariance test
    void testKSSInvariance(KSSICP& kss_icp, const PointCloudPtr& source_raw) {
        std::cout << "\n=== KSS Invariance Test (Similarity Transform) ===" << std::endl;
        PointCloudPtr source_transformed(new PointCloudT);
        Eigen::Vector3f random_translation(0.5f, -0.3f, 0.2f);
        float random_scale = 1.8f;

        source_transformed->reserve(source_raw->size());
        for (const auto& p : *source_raw) {
            PointT pt;
            pt.x = p.x * random_scale + random_translation.x();
            pt.y = p.y * random_scale + random_translation.y();
            pt.z = p.z * random_scale + random_translation.z();
            source_transformed->push_back(pt);
        }

        std::cout << "1. Applied similarity transform: translation(" << random_translation.transpose() << "), scale=" << random_scale << std::endl;

        PointCloudPtr source_pre_raw(new PointCloudT);
        PointCloudPtr source_pre_trans(new PointCloudT);
        Eigen::Vector4f centroid_raw, centroid_trans;
        float scale_raw, scale_trans;

        kss_icp.mapToPreShapeSpace(source_raw, source_pre_raw, centroid_raw, scale_raw);
        kss_icp.mapToPreShapeSpace(source_transformed, source_pre_trans, centroid_trans, scale_trans);

        float dist = kss_icp.computeHausdorffDistance(source_pre_raw, source_pre_trans);
        std::cout << "2. Original/transformed pre-shape point cloud distance: " << dist << std::endl;
        if (dist < 1e-4) {
            std::cout << "KSS invariance verification passed" << std::endl;
        }
        else {
            std::cerr << "KSS invariance verification failed" << std::endl;
        }
        std::cout << "=================================================\n" << std::endl;
    }

} // namespace kss_icp
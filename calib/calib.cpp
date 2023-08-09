#include <iostream>

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#define DLLEXPORT extern "C"

int draw(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcl1, const pcl::PointCloud<pcl::PointXYZ>& pcl2) {

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // pcl::io::loadPCDFile("**./data/bunny.pcd**", *cloud);
    cloud = pcl1;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_milk(new pcl::PointCloud<pcl::PointXYZ>);
    *cloud_milk = pcl2;
    // pcl::io::loadPCDFile("./data/milk_color.pcd", *cloud_milk);

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));

    viewer->setBackgroundColor (0.05, 0.05, 0.05, 0);

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ> (cloud, single_color, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color1(cloud_milk, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ> (cloud_milk, single_color1, "sample cloud milk");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud milk");

    viewer->addCoordinateSystem (0.5);

    while (!viewer->wasStopped()) {
        viewer->spinOnce();
    }
    return 0;
}
double array[18] = {0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,};
DLLEXPORT double* calib(const char * file_name_in, const char * file_name_trans){
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(file_name_in, *cloud_in) == -1) {
        PCL_ERROR("Couldn't read point cloud file\n");
        return array;
    }
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(file_name_trans, *cloud_out) == -1) {
        PCL_ERROR("Couldn't read point cloud file\n");
        return array;
    }
    std::cout<<"load file end: "<<std::endl;
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputCloud(cloud_in);
    icp.setInputTarget(cloud_out);
    // Set the max correspondence distance to 5cm (e.g., correspondences with higher distances will be ignored)
    // icp.setMaxCorrespondenceDistance (1.7976931348623157e+308);
     // Set the maximum number of iterations (criterion 1)
    // icp.setMaximumIterations (1);
    // Set the transformation epsilon (criterion 2)
    // icp.setTransformationEpsilon (2);
    // Set the euclidean distance difference epsilon (criterion 3)
    // icp.setEuclideanFitnessEpsilon (3);
    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final);

    std::cout << "Final " << cloud_in->points.size() << " data points:" << std::endl;
    std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;

    Eigen::Matrix4d transformation_matrix;
    transformation_matrix = icp.getFinalTransformation().cast<double>();
    std::cout << transformation_matrix << std::endl;
    for (int i=0; i<transformation_matrix.rows(); i++){
        for (int j=0; j<transformation_matrix.cols(); j++){
            array[i*4+j] = transformation_matrix(i,j);
        }
    }
    // draw(cloud_out, Final);
    array[16] = icp.hasConverged();
    array[17] = icp.getFitnessScore();
    return array;
}


int main(int argc, char** argv)
{
    std::string file_name_in = "/data/1.pcd";
    std::string file_name_trans = "data/2.pcd";
    calib(file_name_in.c_str(), file_name_trans.c_str());
    std::cout<<"***"<<std::endl;
    return 0;
}

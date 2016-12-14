#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
using namespace cv;
using namespace std;
//int main()
//{
//	char filename[250];
//
//	int max = 2;
//
//
//	vector<Mat> img_buff;
//
//	Mat src, dst;
//	for (int num = 0; num < max; num++){
//		sprintf(filename, "D:/Samsung_Product/result_ver2/original_image.jpg", num);
//		Mat img_temp = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
//		resize(img_temp, dst, Size(50, 50), 0, 0, CV_INTER_NN);
//		img_buff.push_back(dst);
//
//
//
//	}
//	
//		//Mat imgB = imread("C:/Users/davidk/Documents/Visual Studio 2013/Projects/opencv_2_ORB_Feature/Debug/chair_60deg.png", CV_LOAD_IMAGE_GRAYSCALE);
//
//	
//	
//
//	
//	
//	
//	namedWindow("result", CV_WINDOW_AUTOSIZE);
//	imshow("resut", img_buff[0]);
//
//
//
//	waitKey(0);
//
//
//
//
//
//
//	return 0;
//}


////int main(){
////
////	char filename[250];
////
////	int max = 2;
////
////	string dir = "D:/Samsung_Product/result_ver2/";
////	Mat img_buff[2];
////
////	Mat src, dst;
////	Mat img_temp;
////	for (int i = 0; i < max; i++)
////	{
////		if (i == 0){
////			img_buff[0] = imread("D:/Samsung_Product/result_ver2/original_image.jpg", CV_LOAD_IMAGE_GRAYSCALE);
////		}
////		else{
////			img_buff[1] = imread("D:/Samsung_Product/result_ver2/result_0_0_250_case_2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
////
////		}
////		
////		//sprintf(filename, "D:/Samsung_Product/result_ver2/original_image.jpg", num);
////		/*	Mat img_temp = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
////			resize(img_temp, dst, Size(50, 50), 0, 0, CV_INTER_NN);*/
////
////
////
////	}
////
////	
////	
////
////	Mat rotated = imread("D:/Samsung_Product/result_ver2/result_0_0_350_case_1.jpg", 0);
////
////
////	img_buff[0] = img_buff[0].reshape(0, 1); //SINGLE LINE
////	img_buff[1] = img_buff[1].reshape(0, 1); //SINGLE LINE
////
////	//   int size = sizeof(image)/sizeof(Mat);
////	//  image[0].convertTo(image[0], CV_32FC1); //CONVERT TO 32FC1
////	//  image[1].convertTo(image[1], CV_32FC1); //CONVERT TO 32FC1
////	int ii = 0; // Current column in training_mat
////	Mat training_mat;
////	int max_file_num = 2;
////
////	for (int file_num = 0; file_num < max_file_num; file_num++)
////	{ 
////		for (int i = 0; i < img_buff[0].rows; i++) {
////			for (int j = 0; j < img_buff[0].cols; j++) {
////				training_mat.at<float>(file_num, ii++) = img_buff[file_num].at<uchar>(i, j);
////			}
////		}
////	}
//////	Mat new_image(img_buff[0].size().height, 2*img_buff[0].size().width, CV_32FC1, img_buff); //CONVERT TO 32FC1
////
////
////	//Mat labelsmat(max_file_num, 1, CV_32FC1);
////
////	float labels[2] = { 1.0, -1.0 };
////	Mat labelsmat(2, 1, CV_32FC1, labels); //correct labels 1
////
////	labelsmat.convertTo(labelsmat, CV_32FC1);
////
////
////	cout << img_buff[0].size() << endl;
////	//cout << new_image.size() << endl;
////
////
////
////	CvSVMParams params;
////	params.svm_type = CvSVM::C_SVC;
////	params.kernel_type = CvSVM::LINEAR;
////	params.gamma = 3;
////	params.degree = 3;
////	CvSVM svm;
////	svm.train_auto(training_mat, labelsmat, Mat(), Mat(), params);
////
////	//  svm.train_(new_image, labelsmat, Mat(),Mat(),params);
////	//    svm.train(training_mat2, labelsmat, Mat(),Mat(),params);
////
////	// svm.train(training_mat2, labelsmat, Mat(), Mat(), params);
////	svm.save("svm.xml"); // saving
////
////
////	svm.load("svm.xml"); // loading
////
////
////	rotated = rotated.reshape(0, 1);
////	rotated.convertTo(rotated, CV_32FC1);
////
////	cout << svm.predict(rotated) << endl;
////
////}


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>



using namespace cv;
using namespace std;

int main(){
	int num_files = 2;
	

	Mat image[2];
	image[0] = imread("C:/Users/davidk/Documents/Visual Studio 2013/Projects/SVM_with_features/Debug/original_image.jpg", 0);
	image[1] = imread("C:/Users/davidk/Documents/Visual Studio 2013/Projects/SVM_with_features/Debug/result_0_0_300_case_1.jpg", 0);

	
	int width = 500, height = 500;

	resize(image[0], image[0], Size(width, height));
	resize(image[1], image[1], Size(width, height));

	Mat new_image(2, height*width, CV_32FC1); //Training sample from input images

	int ii = 0;
	for (int i = 0; i < num_files; i++){
		Mat temp = image[i];
		ii = 0;
		for (int j = 0; j < temp.rows; j++){
			for (int k = 0; k < temp.cols; k++){
				new_image.at<float>(i, ii++) = temp.at<uchar>(j, k);
			}
		}
	}
	//new_image.push_back(image[0].reshape(0, 1));
	//new_image.push_back(image[1].reshape(0, 1));
	Mat labels(num_files, 1, CV_32FC1);
	labels.at<float>(0, 0) = 1.0;//tomato
	labels.at<float>(1, 0) = -1.0;//melon

	imshow("New image", new_image);
	printf("%f %f", labels.at<float>(0, 0), labels.at<float>(1, 0));

	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.gamma = 3;
	params.degree = 3;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	CvSVM svm;
	svm.train(new_image, labels, Mat(), Mat(), params);

	svm.save("svm.xml"); // saving
	svm.load("svm.xml"); // loading

	Mat test_img = imread("C:/Users/davidk/Documents/Visual Studio 2013/Projects/SVM_with_features/Debug/monitor.jpg", 0);
	resize(test_img, test_img, Size(width, height));
	test_img = test_img.reshape(0, 1);
	//imshow("shit_image", test_img);
	test_img.convertTo(test_img, CV_32FC1);
	float res = svm.predict(test_img);
	if (res > 0)
		cout << endl << "Washer";
	else
		cout << endl << "Not washer";
	waitKey(0);

	return 0;
}
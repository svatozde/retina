// retina.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

// retina.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>


using namespace std;
using namespace cv;

static const cvflann::flann_algorithm_t FLANN_ALGORITHM_INDEX = cvflann::FLANN_INDEX_KDTREE;

struct c_img {
	Mat img;
	std::string id;
	std::string location;
	std::string left_right;
	std::string some;
	std::string measure;
};

struct img_similarity{
	std::string id;
	std::string left_right;
	float score;
};

struct test_results {
	std::string name;
	std::vector<std::pair<c_img, std::vector<img_similarity>>> matches;
};


std::vector<std::string> split(const std::string& str, char delim = '_')
{
	std::vector<std::string> ret;
	std::stringstream ss(str);
	std::string token;
	while (std::getline(ss, token, delim)) {
		ret.push_back(token);
	}
	return ret;
}

std::vector<c_img> getTestData() {
	std::vector<c_img> ret;
	std::string path = "c:\\skola\\MI-ROZ\\SEM\\data\\";
	
	for (auto entry : std::experimental::filesystem::directory_iterator(path)) {
		std::string s_path = entry.path().u8string();
		std::string filename = entry.path().filename().u8string();

		std::vector<std::string> tokens = split(filename);

		Mat img = imread(s_path, IMREAD_COLOR);

		c_img ci = { 
			img, 
			tokens[0],
			tokens[2],
			tokens[3],
			tokens[4],
			tokens[5]
		};
		ret.push_back(ci);
	}
	return ret;
}

class Test {
public:
	test_results run(std::vector<c_img> know, std::vector<c_img> test) { return test_results(); }
};


class NaiveClusterSift : public Test {
public:
	cv::Mat crop(cv::Mat img) {
		cv::Rect myROI(0 + 150, 0 , img.cols - 300, img.rows);
		return img(myROI);
	}


	test_results run(std::vector<c_img> known, std::vector<c_img> test) {
		auto img_clusters = std::map<std::string, std::map<std::string, std::vector<c_img>>>();

		//crop all
		for (auto &img : known) {
			cv::Mat old_img = img.img;
			cv::Mat n_img = crop(img.img);
			~old_img;
			img.img = n_img;
		}

		for (auto &test_img : test) {
			cv::Mat old_img = test_img.img;
			cv::Mat n_img = crop(test_img.img);
			~old_img;
			test_img.img = n_img;
		}

		for (auto &img : known){
			if (img_clusters.find(img.id) == img_clusters.end()) {
				img_clusters[img.id] = std::map<std::string, std::vector<c_img>>();
				img_clusters[img.id]["F"] = std::vector<c_img >();
				img_clusters[img.id]["R"] = std::vector<c_img >();
			}
			img_clusters[img.id][img.left_right].push_back(img);
		}

		

		std::vector<std::pair<c_img, std::vector<img_similarity>>> matches;
		for (auto &test_img : test) {

			std::vector<img_similarity> similarities;
			std::map<std::string, std::map<std::string, std::vector<c_img>>>::iterator it = img_clusters.begin();

			// Iterate over the map using Iterator till end.
			while (it != img_clusters.end())
			{
				std::string id = it->first;

				std::map<std::string, std::vector<c_img>> eye_map = it->second;
				std::map<std::string, std::vector<c_img>>::iterator eye_it = eye_map.begin();

				while (eye_it != eye_map.end()) {
					std::string eye = eye_it->first;
					std::vector<c_img> imgs = eye_it->second;

					int sum = 0;
					for (auto &img : imgs) {
						
						try {
							sum += getScore(test_img.img, img.img);
						}
						catch (const std::exception& e) {
							//cout << e.what() << endl;
						}
					}

					int size = (int)(imgs.size());
					float score = 0;
					if (size > 0) {
						score = sum / (int)(imgs.size());
					}
					cout << "Comparing: " << test_img.id << "_" << test_img.left_right << " with: " << id << "_" << eye << " score: " << score << endl;
					img_similarity similarity = { id, eye, score };
					
					similarities.push_back(similarity);

					eye_it++;
				}

				std::sort(similarities.begin(), similarities.end(), [](const img_similarity &lhs, const img_similarity &rhs) {return lhs.score > rhs.score;});

				std::pair<c_img, std::vector<img_similarity>> all_matches(test_img, similarities);

				matches.push_back(all_matches);

				it++;
			}
		}

		
		
		
		test_results ret = {
			"pure_sift_bf_matcher",
			matches
		};
		return ret;

	}

	int getScore(Mat img1, Mat img2) {
		auto keyPoints1 = std::vector<cv::KeyPoint>();
		auto descriptors1 = cv::Mat();

		auto keyPoints2 = std::vector<cv::KeyPoint>();
		auto descriptors2 = cv::Mat();

		auto siftDetector = cv::xfeatures2d::SIFT::create();

		siftDetector->detectAndCompute(img1, cv::Mat(), keyPoints1, descriptors1);

		siftDetector->detectAndCompute(img2, cv::Mat(), keyPoints2, descriptors2);

		//cout << "# keypoints of image1 :" << keyPoints1.size() << endl;
		//cout << "# keypoints of image2 :" << keyPoints2.size() << endl;


		auto matches12 = std::vector<std::vector<cv::DMatch>>();
		auto matches21 = std::vector<std::vector<cv::DMatch>>();

		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
		matcher->knnMatch(descriptors1, descriptors2, matches12, 2);
		matcher->knnMatch(descriptors1, descriptors2, matches21, 2);

		BFMatcher bfmatcher(NORM_L2, true);
		vector<DMatch> bf_matches;
		bfmatcher.match(descriptors1, descriptors2, bf_matches);
		
		double max_dist = 0; double min_dist = 100;
		for (int i = 0; i < descriptors1.rows; i++)
		{
			double dist = matches12[i].data()->distance;
			if (dist < min_dist)
				min_dist = dist;
			if (dist > max_dist)
				max_dist = dist;
		}

		//printf("-- Max dist : %f \n", max_dist);
		//printf("-- Min dist : %f \n", min_dist);

		//cout << "Matches1-2:" << matches12.size() << endl;
		//cout << "Matches2-1:" << matches21.size() << endl;

		double ratio = 0.9;

		std::vector<DMatch> good_matches1, good_matches2;
		for (int i = 0; i < matches12.size(); i++)
		{
			if (matches12[i].size() >= 2) {
				if (matches12[i][0].distance < 2* min_dist)
					good_matches1.push_back(matches12[i][0]);
			}
		}

		for (int i = 0; i < matches21.size(); i++)
		{
			if (matches21[i].size() >= 2) {
				if (matches21[i][0].distance < 2* min_dist)
					good_matches2.push_back(matches21[i][0]);
			}
		}

		//cout << "Good matches1:" << good_matches1.size() << endl;
		///cout << "Good matches2:" << good_matches2.size() << endl;


		std::vector<DMatch> better_matches;
		int match_count = 0;
		for (int i = 0; i < good_matches1.size(); i++)
		{
			for (int j = 0; j < good_matches2.size(); j++)
			{
				if (good_matches1[i].queryIdx == good_matches2[j].trainIdx && good_matches2[j].queryIdx == good_matches1[i].trainIdx)
				{
					better_matches.push_back(good_matches1[i]);
					match_count++;
					break;
				}
			}
		}

		if (match_count > 3) {
			Mat output;
			drawMatches(img1, keyPoints1, img2, keyPoints2, better_matches, output);
			//imshow("Better matches ",output);
			~output;

			Mat bf_output;
			drawMatches(img1, keyPoints1, img2, keyPoints2, bf_matches, bf_output);
			//imshow("Bruteforce matches ", bf_output);
			~bf_output;

			cout << "bfc: " << bf_matches.size() << " fmatches: " << better_matches.size() << endl;
		}

		return match_count;
	}

};

class PureSift : public Test {
public:
	test_results run(std::vector<c_img> known, std::vector<c_img> test) {
		int counter = 0;
		int failure_counter = 0;
		std::vector<std::pair<c_img, std::vector<img_similarity>>> matches;
		for (auto const& test_img : test) {
			int max = -100;
			c_img best_match;
			for (auto const& know_img : known) {
				counter++;
				try {
					int score = getScore(test_img.img, know_img.img);
					if (score > max) {
						max = score;
						best_match = know_img;
					}
				}
				catch (...) {
					cout << "failure:" << counter << " er:" << ++failure_counter << endl;
				}

			}

			std::pair<std::string, std::string> id(best_match.id, best_match.left_right);
			std::pair<c_img, std::pair<std::string, std::string>>  res(test_img, id);
			//matches.push_back(res);
		}

		test_results ret = {
			"pure_sift_bf_matcher",
			matches
		};
		return ret;
	};

	int getScore(Mat img1, Mat img2) {
		auto keyPoints1 = std::vector<cv::KeyPoint>();
		auto descriptors1 = cv::Mat();

		auto keyPoints2 = std::vector<cv::KeyPoint>();
		auto descriptors2 = cv::Mat();

		auto siftDetector = cv::xfeatures2d::SIFT::create();

		siftDetector->detectAndCompute(img1, cv::Mat(), keyPoints1, descriptors1);

		siftDetector->detectAndCompute(img2, cv::Mat(), keyPoints2, descriptors2);

		//cout << "# keypoints of image1 :" << keyPoints1.size() << endl;
		//cout << "# keypoints of image2 :" << keyPoints2.size() << endl;


		auto matches12 = std::vector<std::vector<cv::DMatch>>();
		auto matches21 = std::vector<std::vector<cv::DMatch>>();

		cv::BFMatcher matcher(cv::NORM_L1, false);
		matcher.knnMatch(descriptors1, descriptors2, matches12, 10);
		matcher.knnMatch(descriptors2, descriptors1, matches21, 10);

		//BFMatcher bfmatcher(NORM_L2, true);
		//vector<DMatch> matches;
		//bfmatcher.match(descriptors1, descriptors2, matches);
		double max_dist = 0; double min_dist = 100;
		for (int i = 0; i < descriptors1.rows; i++)
		{
			double dist = matches12[i].data()->distance;
			if (dist < min_dist)
				min_dist = dist;
			if (dist > max_dist)
				max_dist = dist;
		}

		//printf("-- Max dist : %f \n", max_dist);
		//printf("-- Min dist : %f \n", min_dist);

		//cout << "Matches1-2:" << matches12.size() << endl;
		//cout << "Matches2-1:" << matches21.size() << endl;

		double ratio = 0.9;

		std::vector<DMatch> good_matches1, good_matches2;
		for (int i = 0; i < matches12.size(); i++)
		{
			if (matches12[i].size() >= 2) {
				if (matches12[i][0].distance < ratio * matches12[i][1].distance)
					good_matches1.push_back(matches12[i][0]);
			}
		}

		for (int i = 0; i < matches21.size(); i++)
		{
			if (matches21[i].size() >= 2) {
				if (matches21[i][0].distance < ratio * matches21[i][1].distance)
					good_matches2.push_back(matches21[i][0]);
			}
		}

		cout << "Good matches1:" << good_matches1.size() << endl;
		cout << "Good matches2:" << good_matches2.size() << endl;

		std::vector<DMatch> better_matches;
		int match_count = 0;
		for (int i = 0; i < good_matches1.size(); i++)
		{
			for (int j = 0; j < good_matches2.size(); j++)
			{
				if (good_matches1[i].queryIdx == good_matches2[j].trainIdx && good_matches2[j].queryIdx == good_matches1[i].trainIdx)
				{
					better_matches.push_back(DMatch(good_matches1[i].queryIdx, good_matches1[i].trainIdx, good_matches1[i].distance));
					match_count++;
					break;
				}
			}
		}

		/*
		// show it on an image
		Mat output;
		drawMatches(img1, keyPoints1, img2, keyPoints2, better_matches, output);
		imshow("Matches result", output);
		*/

		return match_count;
	}
};



int main()
{

	std::vector<c_img> data = getTestData();
	std::random_shuffle(data.begin(), data.end());

	vector<c_img>::const_iterator first = data.begin();
	vector<c_img>::const_iterator mid = data.begin() + 15;
	vector<c_img>::const_iterator end = data.end();

	std::vector<c_img> test(first, mid);
	std::vector<c_img> know(mid+1, end);


	NaiveClusterSift sift_test;

	test_results results = sift_test.run(know, test);
	
	waitKey(0); // Wait for a keystroke in the window
	return 0;
}

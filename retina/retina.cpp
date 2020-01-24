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
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <omp.h>
#include <algorithm>
#include <random>

#define THREADS 10


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
	std::string filename;
};

struct img_similarity{
	std::string id;
	std::string left_right;
	float score;
	int best_match;
	cv::Mat best_result;
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

		Mat img = imread(s_path, CV_8UC1);

		c_img ci = { 
			img, 
			tokens[0],
			tokens[2],
			tokens[3],
			tokens[4],
			tokens[5],
			filename
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
	std::map<std::string, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> sift_cache;
	

	NaiveClusterSift() {
		sift_cache = std::map<std::string, std::pair<std::vector<cv::KeyPoint>, cv::Mat>>();
	}

	std::pair<std::vector<cv::KeyPoint>, cv::Mat> getDescriptorsFor(c_img img) {
		std::string key = img.filename;
		if (sift_cache.find(key) == sift_cache.end()) {
			auto siftDetector = cv::xfeatures2d::SIFT::create(750, 5, 0.0002, 60, 1.0);
			auto keyPoints1 = std::vector<cv::KeyPoint>();
			auto descriptors1 = cv::Mat();
			siftDetector->detectAndCompute(img.img, cv::Mat(), keyPoints1, descriptors1);
			std::pair<std::vector<cv::KeyPoint>, cv::Mat> keysAndDescriptors(keyPoints1, descriptors1);
			sift_cache[key] = keysAndDescriptors;
		}
		return sift_cache[key];	
	}
	
	
	double distanceBtwPoints(const cv::Point a, const cv::Point b)
	{
		double xDiff = a.x - b.x;
		double yDiff = a.y - b.y;

		return std::sqrt((xDiff * xDiff) + (yDiff * yDiff));
	}

	void gammaCorrection(Mat& src, Mat& dst, float fGamma){
		unsigned char lut[256];
		for (int i = 0; i < 256; i++){
			lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);
		}

		dst = src.clone();
		const int channels = dst.channels();
		switch (channels){
			case 1:{
				MatIterator_<uchar> it, end;
				for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++) {
					*it = lut[(*it)];
				}
				break;
			}
			case 3:{
				MatIterator_<Vec3b> it, end;
				for (it = dst.begin<Vec3b>(), end = dst.end<Vec3b>(); it != end; it++){
					(*it)[0] = lut[((*it)[0])];
					(*it)[1] = lut[((*it)[1])];
					(*it)[2] = lut[((*it)[2])];
				}
				break;
			}
		}
	}

	cv::Mat crop(cv::Mat img) {
		cv::Rect area(0 + 170, 0 , img.cols - 340, img.rows);
		cv::Mat cut = img(area);
		cv::Mat equalized;

		cv::equalizeHist(cut, equalized);

		//RNG rng(12345);

		//cv::Mat saturated;
		~cut;
		//cut.convertTo(saturated, -1, 1.2, -0.1);
	
		return equalized;
	}

	test_results run(std::vector<c_img> known, std::vector<c_img> test) {
		

		auto img_clusters = std::map<std::string, std::map<std::string, std::vector<c_img>>>();
		
		//crop all
	
		#pragma omp parallel for schedule(static, 10) num_threads(10)
		for (int i = 0; i < known.size(); i++) {
			//cout << "threads " << omp_get_num_threads() << endl;
			auto& img = known[i];
			cv::Mat old_img = img.img;
			cv::Mat n_img = crop(img.img);
			~old_img;
			img.img = n_img;
			getDescriptorsFor(img);
			//cout << "Thread " << omp_get_thread_num() << " has completed iteration " << img.filename << endl;
		}
		#pragma omp parallel for schedule(static, 10) num_threads(10) shared(test)
		for (int i = 0; i < test.size(); i++) {
			auto& test_img = test[i];
			cv::Mat old_img = test_img.img;
			cv::Mat n_img = crop(test_img.img);
			~old_img;
			test_img.img = n_img;
			getDescriptorsFor(test_img);
		}
		
		for (int i = 0; i < known.size(); i++){
			auto& img = known[i];
			if (img_clusters.find(img.id) == img_clusters.end()) {
				img_clusters[img.id] = std::map<std::string, std::vector<c_img>>();
			}

			if (img_clusters[img.id].find(img.left_right) == img_clusters[img.id].end()) {
				img_clusters[img.id][img.left_right] = std::vector<c_img>();
			}
			img_clusters[img.id][img.left_right].push_back(img);
		}
		int missing = 0;
		std::vector<std::pair<c_img, std::vector<img_similarity>>> matches;
		
		//
		for (int ii = 0; ii < test.size(); ii++) {
			std::vector<img_similarity> similarities;
			auto& test_img = test[ii];
			if (img_clusters.find(test_img.id) == img_clusters.end()) {
				if (img_clusters[test_img.id].find(test_img.left_right) == img_clusters[test_img.id].end()) {
					missing++;
					cout << "Category: " << test_img.id << "_" << test_img.left_right << " missing" << endl;
					continue;
				}	
			}
			

			std::map<std::string, std::map<std::string, std::vector<c_img>>>::iterator it = img_clusters.begin();
			while (it != img_clusters.end())
			{			
				std::string id = it->first;
				
				std::map<std::string, std::vector<c_img>> eye_map = it->second;
				std::map<std::string, std::vector<c_img>>::iterator eye_it = eye_map.begin();
				
				while (eye_it != eye_map.end()) {
					std::string eye = eye_it->first;
					std::vector<c_img> imgs = eye_it->second;

					int sum = 0;
					int best_match = -1;
					cv::Mat result_img;
					#pragma omp parallel for schedule(static) num_threads(15)
					for (int iii = 0; iii < imgs.size();iii++ ) {
						auto& img = imgs[iii];
						try {
							bool are_same = test_img.id.compare(img.id) == 0;
							std::pair<int, cv::Mat> result = getScore(test_img, img);
							int tmp_score = result.first;
							
							#pragma ompt critical 
							{
							if (best_match < tmp_score) {
								~result_img;
								result_img = result.second;
								best_match = tmp_score;
								}
							}
							
							sum += tmp_score;
							
						}
						catch (const std::exception& e) {
							//cout << e.what() << endl;
						}
					}

					int size = (int)(imgs.size());
					float score = 0;
					if (size > 0) {
						score = (sum) / (int)(imgs.size());
					}
					cout << "Comparing: " << test_img.id << "_" << test_img.left_right << " with: " << id << "_" << eye << " score: " << score << " best_match: " << best_match << " T: " << omp_get_thread_num() << endl;
					
					img_similarity similarity = { id, eye, score, best_match, result_img };
					similarities.push_back(similarity);

					eye_it++;
				}

				it++;
			}

			std::sort(similarities.begin(), similarities.end(), [](const img_similarity &lhs, const img_similarity &rhs) {return lhs.score > rhs.score;});

			std::pair<c_img, std::vector<img_similarity>> all_matches(test_img, similarities);

			cout << "compared: " << test_img.id << "_" << test_img.left_right << " best_match:" << similarities[0].id << "_" << similarities[0].left_right << " : " << similarities[0].best_match << endl;
 
			cv::imwrite("c:\\skola\\MI-ROZ2\\output\\" + test_img.id + "_" + test_img.left_right + "__" + similarities[0].id + "_" + similarities[0].left_right + ".png", similarities[0].best_result);

			matches.push_back(all_matches);
			test_results tmp = {
				"pure_sift_bf_matcher",
				matches
			};
			calculate_results(tmp, missing);
		}

	
		test_results ret = {
			"pure_sift_bf_matcher",
			matches
		};
		return ret;

	}

	std::pair<int, cv::Mat> getScore(c_img img1, c_img img2) {
		auto kPair1 = getDescriptorsFor(img1);
		
		auto keyPoints1 = kPair1.first;
		auto descriptors1 = kPair1.second;

		auto kPair2 = getDescriptorsFor(img2);

		auto keyPoints2 = kPair2.first;
		auto descriptors2 = kPair2.second;
		
		std::vector<DMatch> bf_matches;
		
		
		std::vector< std::vector<DMatch> > knn_matches;
		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
		matcher->knnMatch(descriptors1, descriptors2, knn_matches, 8);

		//delete matcher;
		//-- Filter matches using the Lowe's ratio test
		const float ratio_thresh = 0.7f;
		std::vector<DMatch> good_matches;
		for (size_t i = 0; i < knn_matches.size(); i++)
		{
			if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
			{
				bf_matches.push_back(knn_matches[i][0]);
			}
		}

		
		//cv::BFMatcher bfmatcher(cv::NORM_L2, true);
		//bfmatcher.match(descriptors1, descriptors2, bf_matches);

		std::vector<DMatch> filtered;

		// copy only positive numbers:
		std::copy_if(bf_matches.begin(), bf_matches.end(), std::back_inserter(filtered), [](DMatch i){return i.distance < 2500;});


		std::vector<DMatch> position_filtered;
		for (auto &match : filtered) {
			auto kp1 = keyPoints1[match.queryIdx];
			auto kp2 = keyPoints2[match.trainIdx];

			double dstnc = distanceBtwPoints(kp1.pt, kp2.pt);
			if (dstnc < 35) {
				position_filtered.push_back(match);
			}		
		}
			
		cv::Mat bf_output;
		drawMatches(
			img1.img, keyPoints1, 
			img2.img, keyPoints2, 
			position_filtered, 
			bf_output, 
			Scalar::all(-1), 
			Scalar::all(-1), 
			std::vector<char>(),
			DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
		);
		//imshow("Bruteforce matches ", bf_output);
		//~bf_output;
		std::pair<int, cv::Mat> ret((int)(position_filtered.size()), bf_output);
		return ret;
	}

	void calculate_results(test_results results, int missing) {
		int imag_count = (int)(results.matches.size());

		int best_count = 0;
		for (auto &img : results.matches) {
			auto tested_image = img.first;
			auto first_image = img.second[0];
			if (tested_image.id.compare(first_image.id) == 0
				&& tested_image.left_right.compare(first_image.left_right) == 0) {
				best_count++;
			}
		}

		cout << "BEST SCORE: " << best_count / imag_count << "  TP: " << best_count << " TC: " << imag_count << " MISSING:" << missing << endl;

		//cout << "MATCH IN FIRST 5 BY SCORE" << endl;

		//cout << "MATCH IN FIRST 10 BY SCORE" << endl;


		best_count = 0;
		for (auto &img : results.matches) {
			auto tested_image = img.first;
			std::sort(img.second.begin(), img.second.end(), [](const img_similarity &lhs, const img_similarity &rhs) {return lhs.best_match > rhs.best_match;});

			auto first_image = img.second[0];
			if (tested_image.id.compare(first_image.id) == 0
				&& tested_image.left_right.compare(first_image.left_right) == 0) {
				best_count++;
			}
		}
		cout << " BEST MATCH: " << best_count / imag_count << " TP: " << best_count << " TC: " << imag_count <<endl;

		//cout << "MATCH IN FIRST 5 BY BEST MATCH" << endl;

		//cout << "MATCH IN FIRST 10 BY BEST MATCH" << endl;


	}

};



int main()
{

	std::vector<c_img> data = getTestData();
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

	shuffle(data.begin(), data.end(), std::default_random_engine(seed));
	

	vector<c_img>::const_iterator first = data.begin();
	vector<c_img>::const_iterator mid = data.begin() + 400;
	vector<c_img>::const_iterator end = data.end();

	std::vector<c_img> test(first, mid);
	std::vector<c_img> know(mid+1, end);

	std::sort(test.begin(), test.end(), [](const c_img &lhs, const c_img &rhs) {return ((lhs.id == rhs.id && lhs.left_right < rhs.left_right) || lhs.id < rhs.id) ;});

	NaiveClusterSift sift_test;

	test_results results = sift_test.run(know, test);
	//calculate_results(results);
	
	waitKey(0); // Wait for a keystroke in the window
	return 0;
}



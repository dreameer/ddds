#include "opencv2/objdetect.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/utility.hpp"

#include "opencv2/videoio/videoio_c.h"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc.hpp"

#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>

using namespace cv;
using namespace std;

string cascadeName = "F:/OpenCV/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml"; //based on your own dic
string nestedCascadeName = "F:/OpenCV/opencv/sources/data/haarcascades/haarcascade_eye.xml";

int flag_face = 0;
int flag_per = 0;
Rect face_rect;
void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale, bool tryflip, bool flag_f );

//hide the local functions in an anon namespace

void help(char** av) {
        cout << "The program captures frames from a video file, image sequence (01.jpg, 02.jpg ... 10.jpg) or camera connected to your computer." << endl
             << "Usage:\n" << av[0] << " <video file, image sequence or device number>" << endl
             << "q,Q,esc -- quit" << endl
             << "space   -- save frame" << endl << endl
             << "\tTo capture from a camera pass the device number. To find the device number, try ls /dev/video*" << endl
             << "\texample: " << av[0] << " 0" << endl
             << "\tYou may also pass a video file instead of a device number" << endl
             << "\texample: " << av[0] << " video.avi" << endl
             << "\tYou can also pass the path to an image sequence and OpenCV will treat the sequence just like a video." << endl
             << "\texample: " << av[0] << " right%%02d.jpg" << endl;
    }

int process(VideoCapture& capture) {
        int n = 0;
        char filename[200];
		double t;
        string window_name = "电子科大学-彭强 | press d to detect,q or esc to quit";
        cout << "press space to save a picture. q or esc to quit" << endl;
        namedWindow(window_name, WINDOW_KEEPRATIO); //resizable window;
        Mat frame;
		
		CascadeClassifier cascade,nestedCascade;
		double scale = 1.3;
		bool tryflip = false;
		if( !nestedCascade.load( nestedCascadeName ) )
			//cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
		    cout << "wrong with nestedCascade" <<endl;
		if( !cascade.load( cascadeName ) )
		{
			//cerr << "ERROR: Could not load classifier cascade" << endl;
			//help();
			//return -1;
			cout << "wrong with cascade" << endl;
		}
        for (;;) {
            capture >> frame;
			cout<<frame.cols<<","<<frame.rows<<endl;
            if (frame.empty())
                break;

            imshow(window_name, frame);
            char key = (char)waitKey(30); //delay N millis, usually long enough to display and capture input

            switch (key) {
			case 'd':
				if( !nestedCascade.load( nestedCascadeName ) )
					//cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
				    cout << "wrong with nestedCascade" <<endl;
				if( !cascade.load( cascadeName ) )
				{
					//cerr << "ERROR: Could not load classifier cascade" << endl;
					//help();
					//return -1;
					cout << "wrong with cascade" << endl;
				}
				detectAndDraw( frame, cascade, nestedCascade, scale, tryflip, true);
				break;
            case 'q':
            case 'Q':
            case 27: //escape key
                return 0;
            case ' ': //Save an image
                sprintf(filename,"filename%.3d.jpg",n++);
                imwrite(filename,frame);
                cout << "Saved " << filename << endl;
                break;
            default:
				t = (double)cvGetTickCount();
				if(1 == flag_face){
					if(0<face_rect.x&&0<face_rect.width&&face_rect.x+face_rect.width<frame.cols&&0<face_rect.y&&0<face_rect.height&&face_rect.y+face_rect.height<frame.rows){
						Mat face = frame(face_rect);
					    namedWindow( "face_rect", WINDOW_NORMAL );
                        imshow( "face_rect", face );
					    detectAndDraw(face, cascade, nestedCascade, scale, tryflip, false );
					}
					else
						detectAndDraw( frame, cascade, nestedCascade, scale, tryflip, true );
				}
				else
					detectAndDraw( frame, cascade, nestedCascade, scale, tryflip, true );
				t = (double)cvGetTickCount() - t;
				printf( "all time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
                break;
            }
        }
        return 0;
    }


void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale, bool tryflip, bool flag_f )
{

	int i = 0;
    double t = 0;
	double t_ = 0;
    vector<Rect> faces, faces2;
    const static Scalar colors[] =  { CV_RGB(0,0,255),
        CV_RGB(0,128,255),
        CV_RGB(0,255,255),
        CV_RGB(0,255,0),
        CV_RGB(255,128,0),
        CV_RGB(255,255,0),
        CV_RGB(255,0,0),
        CV_RGB(255,0,255)} ;
    Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );

    cvtColor( img, gray, COLOR_BGR2GRAY );
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );

    t = (double)cvGetTickCount();
    cascade.detectMultiScale( smallImg, faces,
        1.1, 16, 0
        //|CASCADE_FIND_BIGGEST_OBJECT
        //|CASCADE_DO_ROUGH_SEARCH
        |CASCADE_SCALE_IMAGE
        ,
        Size(100, 100) );
    if( tryflip )
    {
        flip(smallImg, smallImg, 1);
        cascade.detectMultiScale( smallImg, faces2,
                                 1.1, 4, 0
                                 //|CASCADE_FIND_BIGGEST_OBJECT
                                 //|CASCADE_DO_ROUGH_SEARCH
                                 |CASCADE_SCALE_IMAGE
                                 ,
                                 Size(300, 300) );
        for( vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); r++ )
        {
            faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
        }
    }
    t_ = (double)cvGetTickCount() - t;
    printf( "face detection time = %g ms\n", t_/((double)cvGetTickFrequency()*1000.) );
	printf("faces.size=%d\n",faces.size());
	if(faces.size()==1){
		cout<<faces[0].x<<" "<<faces[0].y<<" "<<faces[0].width<<" "<<faces[0].height<<endl;
		flag_face = 1;
		double width_new = (cvRound((faces[0].width + faces[0].height)*0.25*scale))*2.4;
		if(flag_f){
			int ra = cvRound((faces[0].width + faces[0].height)*0.25*scale);
			face_rect.x = cvRound((faces[0].x + faces[0].width*0.5)*scale) - ra*1.2;
            face_rect.y = cvRound((faces[0].y + faces[0].height*0.5)*scale) - ra*1.2;
			face_rect.width = ra*2.4;
		    face_rect.height = ra*2.4;
		    cout<<face_rect.x<<" "<<face_rect.y<<" "<<face_rect.width<<" "<<face_rect.height<<endl;
		}
		else if((double)width_new/(double)face_rect.width<0.6){
			int ra = cvRound((faces[0].width + faces[0].height)*0.25*scale);
			face_rect.x = cvRound((faces[0].x + faces[0].width*0.5)*scale) - ra*1.2;
            face_rect.y = cvRound((faces[0].y + faces[0].height*0.5)*scale) - ra*1.2;
			face_rect.width = ra*2.4;
		    face_rect.height = ra*2.4;
		    cout<<face_rect.x<<" "<<face_rect.y<<" "<<face_rect.width<<" "<<face_rect.height<<endl;
		}
	}
	else
		flag_face = 0;
	t_ = (double)cvGetTickCount() - t;
    printf( "detection time = %g ms\n", t_/((double)cvGetTickFrequency()*1000.) );
    for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
    {
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = colors[i%8];
        int radius;
        double aspect_ratio = (double)r->width/r->height;
        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
        {
            center.x = cvRound((r->x + r->width*0.5)*scale);
            center.y = cvRound((r->y + r->height*0.5)*scale);
            radius = cvRound((r->width + r->height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }
        else
            rectangle( img, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
                       cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
                       color, 3, 8, 0);
        if( nestedCascade.empty() )
            continue;
        smallImgROI = smallImg(*r);
        nestedCascade.detectMultiScale( smallImgROI, nestedObjects,
            1.1, 16, 0
            //|CASCADE_FIND_BIGGEST_OBJECT
            //|CASCADE_DO_ROUGH_SEARCH
            //|CASCADE_DO_CANNY_PRUNING
            |CASCADE_SCALE_IMAGE
            ,
            Size(30, 30) );
		t_ = (double)cvGetTickCount() - t;
        printf( "eye detection time = %g ms\n", t_/((double)cvGetTickFrequency()*1000.) );
        for( vector<Rect>::const_iterator nr = nestedObjects.begin(); nr != nestedObjects.end(); nr++ )
        {
            center.x = cvRound((r->x + nr->x + nr->width*0.5)*scale);
            center.y = cvRound((r->y + nr->y + nr->height*0.5)*scale);
            radius = cvRound((nr->width + nr->height)*0.25*scale);
            //circle( img, center, radius, color, 3, 8, 0 );
        }
		
		//Mat eye(img,Rect(center.x-radius,center.y-radius,2*radius,2*radius));
		Mat eye(img,Rect(center.x-30,center.y-20,60,40));

        namedWindow( "oringin_eye", WINDOW_AUTOSIZE );
        imshow( "oringin_eye", eye );


		//caculate the area
		  Mat src_gray;
		  Mat threshold_output;
          vector<vector<Point> > contours;
          vector<Vec4i> hierarchy;
		  RNG rng(12345);
		  int eye_flag = 0; //open is 1,closed is 0;
          /// Convert image to gray and blur it
          cvtColor( eye, src_gray, COLOR_BGR2GRAY );
          blur( src_gray, src_gray, Size(3,3) );
          

          /// Detect edges using Threshold
          threshold( src_gray, threshold_output, 40, 255, 1 );
		  namedWindow( "threshold", WINDOW_AUTOSIZE );
          imshow( "threshold", threshold_output );

          /// Find contours
          findContours( threshold_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

          /// Approximate contours to polygons + get bounding rects and circles
          vector<vector<Point> > contours_poly( contours.size() );
          vector<Rect> boundRect( contours.size() );

          for( size_t i = 0; i < contours.size(); i++ )
		  { 
			 approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
             boundRect[i] = boundingRect( Mat(contours_poly[i]) );
		  }
		  Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );

		  eye_flag = 2; //meas eye closed;
		  for( size_t i = 0; i< contours.size(); i++ )
          {
			  if(boundRect[i].width/boundRect[i].height<4)
				  if(boundRect[i].height>4){
					  Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
                      drawContours( drawing, contours_poly, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point() );
                      rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
				      eye_flag = 1;
				      cout<<"openn degree="<<boundRect[i].height<<endl;
                      //circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
				  }
		  }
          t_ = (double)cvGetTickCount() - t;
          printf( "eye state detection time = %g ms\n", t_/((double)cvGetTickFrequency()*1000.) );

          namedWindow( "Contours", WINDOW_AUTOSIZE );
          imshow( "Contours", drawing );
          
		  Point text_location;
		  text_location.x=10;
		  text_location.y=40;

		  if(2==eye_flag){
			  flag_per = flag_per + 1;
			  if(flag_per>5){
				  putText( img, "you are tired !", text_location, CV_FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 2, 8 );
		          namedWindow( "pengqiang_face_plus_eye", WINDOW_AUTOSIZE );
                  imshow( "pengqiang_face_plus_eye", img );
			  }
		  }
		  else
			  flag_per = 0;
	      //putText( img, "eye open", text_location, CV_FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 2, 8 );
		  //namedWindow( "pengqiang_face_plus_eye", WINDOW_AUTOSIZE );
          //imshow( "pengqiang_face_plus_eye", img );
		
    }
}

int main(int ac, char** av) {
if (ac != 2) {
        help(av);
        return 1;
    }
    std::string arg = av[1];
    VideoCapture capture(arg); //try to open string, this will attempt to open it as a video file or image sequence
    if (!capture.isOpened()) //if this fails, try to open as a video camera, through the use of an integer param
        capture.open(atoi(arg.c_str()));
    if (!capture.isOpened()) {
        cerr << "Failed to open the video device, video file or image sequence!\n" << endl;
        help(av);
        return 1;
    }
    return process(capture);
}


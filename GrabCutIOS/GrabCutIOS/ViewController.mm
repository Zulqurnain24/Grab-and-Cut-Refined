//
//  ViewController.m
//  Grab the object in focus
//
//  Created by Zulqurnain on 2017. 4. 18..
//  Copyright (c) Truffle technologies.
//  @Author Mohammad Zulqurnain
//

#import "ViewController.h"
#import "GrabCutManager.h"
#import "TouchDrawView.h"
#import <MobileCoreServices/UTCoreTypes.h>

static inline double radians (double degrees) {return degrees * M_PI/180;}
static int MAX_IMAGE_LENGTH = 450;
int padding = 1;
@interface ViewController ()<UINavigationControllerDelegate, UIImagePickerControllerDelegate>
@property (weak, nonatomic) IBOutlet UIImageView *imageView;
@property (weak, nonatomic) IBOutlet UIImageView *resultImageView;
@property (nonatomic) CGPoint startPoint;
@property (nonatomic) CGPoint endPoint;
@property (weak, nonatomic) IBOutlet TouchDrawView *touchDrawView;
@property (nonatomic, strong) GrabCutManager* grabcut;
@property (weak, nonatomic) IBOutlet UILabel *stateLabel;
@property (weak, nonatomic) IBOutlet UIButton *rectButton;

@property (weak, nonatomic) IBOutlet UIButton *plusButton;

@property (weak, nonatomic) IBOutlet UIButton *minusButton;

@property (weak, nonatomic) IBOutlet UIButton *doGrabcutButton;

@property (weak, nonatomic) IBOutlet UIButton *doSaveToPhotorollButton;
@property (weak, nonatomic) IBOutlet UIButton *openPhotoRollButton;
@property (weak, nonatomic) IBOutlet UIButton *openCameraButton;

@property (nonatomic, assign) TouchState touchState;
@property (nonatomic, assign) CGRect grabRect;
@property (nonatomic, strong) UIImage* originalImage;
@property (nonatomic, strong) UIImage* resizedImage;
@property (nonatomic, strong) UIImagePickerController* imagePicker;

@property (nonatomic) UIActivityIndicatorView *spinner;
@property (nonatomic) UIView* dimmedView;

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    CGFloat width = [UIScreen mainScreen].bounds.size.width;
    MAX_IMAGE_LENGTH = width;
    // Do any additional setup after loading the view, typically from a nib.
    _grabcut = [[GrabCutManager alloc] init];
    
    _originalImage = [UIImage imageNamed:@"test.jpg"];
    _resizedImage = [self getProperResizedImage:_originalImage];
    
    [self initStates];
    
    _doGrabcutButton.layer.cornerRadius = 15;
    _doGrabcutButton.clipsToBounds = true;
    _doSaveToPhotorollButton.layer.cornerRadius = 15;
    _doSaveToPhotorollButton.clipsToBounds = true;
    _openPhotoRollButton.layer.cornerRadius = 15;
    _openPhotoRollButton.clipsToBounds = true;
    _openCameraButton.layer.cornerRadius = 15;
    _openCameraButton.clipsToBounds = true;
    _rectButton.layer.cornerRadius = 15;
    _rectButton.clipsToBounds = true;
}

-(void) initStates{
    _touchState = TouchStateNone;
    [self updateStateLabel];
    
    _rectButton.enabled = YES;
    _plusButton.enabled = NO;
    _minusButton.enabled = NO;
    _doGrabcutButton.enabled = NO;
    
}


-(void) doGrabcut{
    
    
    //    _touchState = TouchStateRect;
    //    [self updateStateLabel];
    //
    //    _plusButton.enabled = NO;
    //    _minusButton.enabled = NO;
    //    _doGrabcutButton.enabled = YES;
    //
    //    NSLog(@"began");
    
    
    //    CGFloat height = [UIScreen mainScreen].bounds.size.height;
    //    self.startPoint = CGPointMake(0, 0);
    //    self.endPoint = CGPointMake(width , height );
    //    CGRect rect = [self getTouchedRect:self.startPoint endPoint:self.endPoint];
    //    [self.touchDrawView drawRectangle:rect];
    //    _grabRect = [self getTouchedRectWithImageSize:_resizedImage.size];
    
    [self showLoadingIndicatorView];
    __weak typeof(self)weakSelf = self;
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT,
                                             (unsigned long)NULL), ^(void) {
        /**
         # opencv image processing part #
         */
        
        cv::Mat mat = [self cvMatWithImage:weakSelf.resizedImage ];
        
        cvtColor(mat, mat, CV_BGR2RGB);
        
        cv::Mat imageToBeProcessed = mat.clone();
        
        fastNlMeansDenoisingColored(mat, mat);
        mat = _filter_color_enhance(mat);
       
        cv::Mat kernel = (Mat_<float>(3,3) <<
                      0,  -1, 0,
                      -1, 5, -1,
                      0,  -1, 0);
    
        filter2D(mat, mat , -1, kernel);
        filter2D(mat, mat , -1, kernel);
        // first, the good result
        //Laplacian(imageToBeProcessed, imageToBeProcessed, CV_8UC1);
        
        int scale = 1;
        int delta = 0;
        int ddepth = CV_16S;
        /// Generate grad_x and grad_y
        Mat grad_x, grad_y;
        Mat abs_grad_x, abs_grad_y;
        Mat grad;
        /// Gradient X
        //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
        Sobel( imageToBeProcessed, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
        convertScaleAbs( grad_x, abs_grad_x );
        
        /// Gradient Y
        //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
        Sobel( imageToBeProcessed, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
        convertScaleAbs( grad_y, abs_grad_y );
        /// Total Gradient (approximate)
        addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
       
//        cv::GaussianBlur(mat, mat, cv::Size(0, 0), 3);
//        cv::addWeighted(mat, 1.5, mat, -0.5, 0, mat);

//        mat = mat + grad;

        mat =  (0.5 * mat) +  (0.5 * grad);
        
        UIImage* image = [self UIImageFromCVMat:mat];
        //
        UIImage* resultImage = [weakSelf.grabcut doGrabCut:image foregroundBound:weakSelf.grabRect iterationCount:10];
        //resultImage = [weakSelf.grabcut doGrabCut:resultImage foregroundBound:weakSelf.grabRect iterationCount:10];
        resultImage = [weakSelf masking:weakSelf.originalImage mask:[weakSelf resizeImage:resultImage size:weakSelf.originalImage.size]];
        
        mat = [self cvMatWithImage:resultImage ];
        
        // load as color image BGR
        cv::Mat input = [self cvMatWithImage:[UIImage imageNamed:@"white.png"] ];
        
        cv::Mat input_bgra = mat;
        cv::cvtColor(input, mat, CV_BGR2BGRA);
        
        // find all white pixel and set alpha value to zero:
        for (int y = 0; y < input_bgra.rows; ++y)
            for (int x = 0; x < input_bgra.cols; ++x)
            {
                cv::Vec4b & pixel = input_bgra.at<cv::Vec4b>(y, x);
                // if pixel is white
                if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0)
                {
                    // set alpha to zero:
                    pixel[0] = 255;
                    pixel[1] = 255;
                    pixel[2] = 255;
                    pixel[3] = 255;
                    
                }
            }
        cv::Mat copyImage = input_bgra.clone();
        cv::GaussianBlur(copyImage, copyImage, cv::Size(7,7), 7);
        scale = 1;
        delta = 0;
        ddepth = CV_16S;

        /// Gradient X
        //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
        Sobel( copyImage, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
        convertScaleAbs( grad_x, abs_grad_x );
        
        /// Gradient Y
        //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
        Sobel( copyImage, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
        convertScaleAbs( grad_y, abs_grad_y );
        
        /// Total Gradient (approximate)
        addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, copyImage );
        
        image = [self UIImageFromCVMat:copyImage];
        [weakSelf.imageView setImage:image];
        [weakSelf.imageView setAlpha:0.3];
        cv::Mat fadedMat = [self cvMatWithImage:[weakSelf.imageView image]];
        
        copyImage = input_bgra - fadedMat ;
    
        resultImage = [self UIImageFromCVMat:copyImage];
       
        dispatch_async(dispatch_get_main_queue(), ^(void) {
            [weakSelf.resultImageView setImage:resultImage];
            [weakSelf.resultImageView setAlpha:0.70];
            [weakSelf.imageView setImage:[self UIImageFromCVMat:input_bgra]];
            [weakSelf.imageView setAlpha:1.0];
            [weakSelf hideLoadingIndicatorView];
        });
    });
}




-(UIImage *) getProperResizedImage:(UIImage*)original{
    float ratio = original.size.width/original.size.height;
    
    if(original.size.width > original.size.height){
        if(original.size.width > MAX_IMAGE_LENGTH){
            return [self resizeWithRotation:original size:CGSizeMake(MAX_IMAGE_LENGTH, MAX_IMAGE_LENGTH/ratio)];
        }
    }else{
        if(original.size.height > MAX_IMAGE_LENGTH){
            return [self resizeWithRotation:original size:CGSizeMake(MAX_IMAGE_LENGTH*ratio, MAX_IMAGE_LENGTH)];
        }
    }
    
    return original;
}

-(NSString*) getTouchStateToString{
    NSString* state = @"Touch State : ";
    NSString* suffix;
    
    switch (_touchState) {
        case TouchStateNone:
            suffix = @"None";
            break;
        case TouchStateRect :
            suffix = @"Rect";
            break;
        case TouchStatePlus :
            suffix = @"Plus";
            break;
        case TouchStateMinus :
            suffix = @"Minus";
            break;
        default:
            break;
    }
    
    return [NSString stringWithFormat:@"%@%@", state, suffix];
}

-(void) updateStateLabel{
    [self.stateLabel setText:[self getTouchStateToString]];
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

-(CGRect) getTouchedRectWithImageSize:(CGSize) size{
    CGFloat widthScale = size.width/self.imageView.frame.size.width;
    CGFloat heightScale = size.height/self.imageView.frame.size.height;
    return [self getTouchedRect:_startPoint endPoint:_endPoint widthScale:widthScale heightScale:heightScale];
}

-(CGRect) getTouchedRect:(CGPoint)startPoint endPoint:(CGPoint)endPoint{
    return [self getTouchedRect:startPoint endPoint:endPoint widthScale:1.0 heightScale:1.0];
}

-(CGRect) getTouchedRect:(CGPoint)startPoint endPoint:(CGPoint)endPoint widthScale:(CGFloat)widthScale heightScale:(CGFloat)heightScale{
    CGFloat minX = startPoint.x > endPoint.x ? endPoint.x*widthScale : startPoint.x*widthScale;
    CGFloat maxX = startPoint.x < endPoint.x ? endPoint.x*widthScale : startPoint.x*widthScale;
    CGFloat minY = startPoint.y > endPoint.y ? endPoint.y*heightScale : startPoint.y*heightScale;
    CGFloat maxY = startPoint.y < endPoint.y ? endPoint.y*heightScale : startPoint.y*heightScale;
    
    return CGRectMake(minX, minY, maxX - minX, maxY - minY);
}

-(UIImage*) resizeImage:(UIImage*)image size:(CGSize)size{
    UIGraphicsBeginImageContext(size);
    CGContextRef context = UIGraphicsGetCurrentContext();
    CGContextTranslateCTM(context, 0.0, size.height);
    CGContextScaleCTM(context, 1.0, -1.0);
    
    CGContextDrawImage(context, CGRectMake(0.0, 0.0, size.width, size.height), [image CGImage]);
    UIImage *scaledImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    
    return scaledImage;
}

-(UIImage*) resizeWithRotation:(UIImage *) sourceImage size:(CGSize) targetSize
{
    CGFloat targetWidth = targetSize.width;
    CGFloat targetHeight = targetSize.height;
    
    CGImageRef imageRef = [sourceImage CGImage];
    CGBitmapInfo bitmapInfo = CGImageGetBitmapInfo(imageRef);
    CGColorSpaceRef colorSpaceInfo = CGImageGetColorSpace(imageRef);
    
    if (bitmapInfo == kCGImageAlphaNone) {
        bitmapInfo = kCGImageAlphaNoneSkipLast;
    }
    
    CGContextRef bitmap;
    
    if (sourceImage.imageOrientation == UIImageOrientationUp || sourceImage.imageOrientation == UIImageOrientationDown) {
        bitmap = CGBitmapContextCreate(NULL, targetWidth, targetHeight, CGImageGetBitsPerComponent(imageRef), CGImageGetBytesPerRow(imageRef), colorSpaceInfo, bitmapInfo);
        
    } else {
        bitmap = CGBitmapContextCreate(NULL, targetHeight, targetWidth, CGImageGetBitsPerComponent(imageRef), CGImageGetBytesPerRow(imageRef), colorSpaceInfo, bitmapInfo);
        
    }
    
    if (sourceImage.imageOrientation == UIImageOrientationLeft) {
        CGContextRotateCTM (bitmap, radians(90));
        CGContextTranslateCTM (bitmap, 0, -targetHeight);
        
    } else if (sourceImage.imageOrientation == UIImageOrientationRight) {
        CGContextRotateCTM (bitmap, radians(-90));
        CGContextTranslateCTM (bitmap, -targetWidth, 0);
        
    } else if (sourceImage.imageOrientation == UIImageOrientationUp) {
        // NOTHING
    } else if (sourceImage.imageOrientation == UIImageOrientationDown) {
        CGContextTranslateCTM (bitmap, targetWidth, targetHeight);
        CGContextRotateCTM (bitmap, radians(-180.));
    }
    
    CGContextDrawImage(bitmap, CGRectMake(0, 0, targetWidth, targetHeight), imageRef);
    CGImageRef ref = CGBitmapContextCreateImage(bitmap);
    UIImage* newImage = [UIImage imageWithCGImage:ref];
    
    CGContextRelease(bitmap);
    CGImageRelease(ref);
    
    return newImage; 
}

-(UIImage *) masking:(UIImage*)sourceImage mask:(UIImage*) maskImage{
    //Mask Image
    CGImageRef maskRef = maskImage.CGImage;
    
    CGImageRef mask = CGImageMaskCreate(CGImageGetWidth(maskRef),
                                        CGImageGetHeight(maskRef),
                                        CGImageGetBitsPerComponent(maskRef),
                                        CGImageGetBitsPerPixel(maskRef),
                                        CGImageGetBytesPerRow(maskRef),
                                        CGImageGetDataProvider(maskRef), NULL, false);
    
    CGImageRef masked = CGImageCreateWithMask([sourceImage CGImage], mask);
    CGImageRelease(mask);
    
    UIImage *maskedImage = [UIImage imageWithCGImage:masked];
    
    CGImageRelease(masked);
    
    return maskedImage;
}

-(CGSize) getResizeForTimeReduce:(UIImage*) image{
    CGFloat ratio = image.size.width/ image.size.height;
    
    if([image size].width > [image size].height){

        
        if(image.size.width > 400){
            return CGSizeMake(400, 400/ratio);
        }else{
            return image.size;
        }
        
    }else{
        if(image.size.height > 400){
            return CGSizeMake(ratio/400, 400);
        }else{
            return image.size;
        }
    }
}


- (IBAction)tapOnRect:(id)sender {
    _touchState = TouchStateRect;
    [self updateStateLabel];
    
    _plusButton.enabled = NO;
    _minusButton.enabled = NO;
    _doGrabcutButton.enabled = YES;
    
    NSLog(@"began");
    CGFloat width = [UIScreen mainScreen].bounds.size.width;
    CGFloat height = [UIScreen mainScreen].bounds.size.height;
    self.startPoint = CGPointMake(padding, padding);
    self.endPoint = CGPointMake(width - padding, height - padding);
    CGRect rect = [self getTouchedRect:self.startPoint endPoint:self.endPoint];
    [self.touchDrawView drawRectangle:rect];
    _grabRect = [self getTouchedRectWithImageSize:_resizedImage.size];
    cv::Mat img = [self cvMatWithImage:_imageView.image];
    _grabRect = [self HighlightContours: img];
    self.startPoint = CGPointMake(_grabRect.origin.x, _grabRect.origin.y);
    self.endPoint = CGPointMake(_grabRect.origin.x + _grabRect.size.width, _grabRect.origin.y + _grabRect.size.height);
   
}


-(CGRect)HighlightContours:(cv::Mat&)imgSrc {
    
    double threshold = 128; // needs adjustment.
    int n_erode_dilate = 1; // needs adjustment.
    
    cv::Mat m = imgSrc.clone();
    cv::cvtColor(m, m, CV_RGB2GRAY); // convert to glayscale image.
    cv::blur(m, m, cv::Size(5,5));
    cv::threshold(m, m, threshold, 255,CV_THRESH_BINARY_INV);
    cv::erode(m, m, cv::Mat(),cv::Point(-1,-1),n_erode_dilate);
    cv::dilate(m, m, cv::Mat(),cv::Point(-1,-1),n_erode_dilate);
    
    std::vector< std::vector<cv::Point> > contours;
    std::vector<cv::Point> points;
    cv::findContours(m, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
    for (size_t i=0; i<contours.size(); i++) {
        for (size_t j = 0; j < contours[i].size(); j++) {
            cv::Point p = contours[i][j];
            points.push_back(p);
        }
    }
    // And process the points or contours to pick up specified object.
    
    // for example: draws rectangle on original image.
    if(points.size() > 0){
        cv::Rect brect = cv::boundingRect(cv::Mat(points).reshape(2));
        cv::rectangle(imgSrc, brect.tl(), brect.br(), cv::Scalar(255, 255, 255), 2, CV_AA);
        CGRect rect = CGRectMake(brect.x, brect.y, brect.width, brect.height);
        return rect;
    }
    CGRect rect = CGRectMake(0, 0, imgSrc.cols, imgSrc.rows);
    return rect;
 
}

-(UIImage *)UIImageFromCVMat:(cv::Mat)cvMat {
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    
    CGColorSpaceRef colorSpace;
    CGBitmapInfo bitmapInfo;
    
    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
        bitmapInfo = kCGImageAlphaNone | kCGBitmapByteOrderDefault;
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
        bitmapInfo = kCGBitmapByteOrder32Little | (
                                                   cvMat.elemSize() == 3? kCGImageAlphaNone : kCGImageAlphaNoneSkipFirst
                                                   );
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(
                                        cvMat.cols,                 //width
                                        cvMat.rows,                 //height
                                        8,                          //bits per component
                                        8 * cvMat.elemSize(),       //bits per pixel
                                        cvMat.step[0],              //bytesPerRow
                                        colorSpace,                 //colorspace
                                        bitmapInfo,                 // bitmap info
                                        provider,                   //CGDataProviderRef
                                        NULL,                       //decode
                                        false,                      //should interpolate
                                        kCGRenderingIntentDefault   //intent
                                        );
    
    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return finalImage; 
}

-(cv::Mat)cvMatWithImage:(UIImage *)image
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to backing data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    (kCGBitmapByteOrder32Host | kCGImageAlphaPremultipliedFirst) |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    
    return cvMat;
}


/**
 * track the card, crop, and rotate the card
 * @param img_name_str name of the card, mainly for logging
 * @param raw_card original input card (from Mr.Mohammed, extracted from video frame)
 * @return tracked card, or just empty cv::Mat incase of none if found
 */
cv::Mat track_card(cv::Mat &raw_card) {
    // cv::Mat1b src;
    cvtColor(raw_card, raw_card, CV_HSV2BGR);
    
    cv::cvtColor(raw_card, raw_card, cv::COLOR_BGR2GRAY);
    
    cv::Mat thr;
    
    cv::threshold(raw_card, thr, _get_max_histogram(raw_card) * 23 / 25, 255, cv::THRESH_BINARY);
    
    int largest_area=0;
    int largest_contour_index=0;
    cv::Rect bounding_rect;
    
    cv::Mat dst(thr.rows,thr.cols,CV_8UC1,cv::Scalar::all(0));
    
    std::vector<std::vector<cv::Point> > contours; // Vector for storing contour
    std::vector<cv::Vec4i> hierarchy;
    
    
    findContours( thr, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE ); // Find the contours in the image
    
    
    
    if (!contours.empty()) {
        
        for( int i = 0; i< contours.size(); i++ ) // iterate through each contour.
        {
            double a=contourArea( contours[i],false);  //  Find the area of contour
            if(a>largest_area){
                largest_area=a;
                largest_contour_index=i;                //Store the index of largest contour
                bounding_rect=boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
                NSLog(@"largest contour %f", a);
                NSLog(@"contour size: %lu", contours[i].size());
                NSLog(@"contour width: %d height: %d", bounding_rect.width, bounding_rect.height);
                
            }
            
        }
        
        cv::Scalar color( 255,255,255);
        drawContours( dst, contours,largest_contour_index, color, CV_FILLED, 8, hierarchy ); // Draw the largest contour using previously stored index.
        
        //        /* rotate and crop*/
        cv::RotatedRect rect = cv::minAreaRect(contours[largest_contour_index]);
        cv::Mat M, rotated, cropped;
        float angle = rect.angle;
        cv::Size rect_size = rect.size;
        if (rect.angle < -45.) {
            angle += 90.0;
            int tempDim = rect_size.width;
            rect_size.width = rect_size.height;
            rect_size.height = tempDim;
        }
        
        M = cv::getRotationMatrix2D(rect.center, angle, 1.25);
        cv::warpAffine(dst, rotated, M, dst.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
        cv::getRectSubPix(rotated, rect_size, rect.center, cropped);
        
        NSLog(@"cropped");
        
        return cropped;
        
    } else {
        
        return dst;
    }
}

/**
 * get max histogram index
 * @param img
 * @return index of color (or gray-scaled) that have max histogram
 */
static int _get_max_histogram(const cv::Mat& img) {
    // Check if image is empty or gray-scaled?
    if (img.empty()) {
        return 0;
    }
    
    cv::Mat src;
    if (img.channels() > 1) {
        cv::cvtColor(img, src, cv::COLOR_BGR2GRAY);
    } else {
        img.copyTo(src);
    }
    
    // Establish the number of bins
    int h_histSize = 256;
    
    // Set the ranges ( for B,G,R) )
    float h_range[] = {0, 255};
    const float* h_histRange = {h_range};
    
    cv::Mat h_hist;
    
    // Compute the histograms:
    calcHist(&src, 1, 0, cv::Mat(), h_hist, 1, &h_histSize, &h_histRange);
    
    // Normalize the result to [ 0, histImage.rows ]
    cv::Mat histImage(h_histSize, h_histSize, CV_8UC3, cv::Scalar(0, 0, 0));
    normalize(h_hist, h_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    
    int max = 0;
    int max_index = 0;
    for (int i = 1; i < h_histSize; i++) {
        if (max < h_hist.at<float>(i)) {
            max = cvRound(h_hist.at<float>(i));
            max_index = i;
        }
    }
    
    return max_index;
}

static cv::Mat _filter_color_enhance(const cv::Mat& img) {
    cv::Mat hsv_image, channels[3];
    cvtColor(img, hsv_image, cv::COLOR_BGR2HSV);
    split(hsv_image, channels);
    
    uchar max_s = 0;
    uchar min_s = 255;
    for (auto y = 0; y < channels[1].rows; y++) {
        for (auto x = 0; x < channels[1].cols; x++) {
            uchar v = channels[1].at<uchar>(y, x);
            if (v > max_s) max_s = v;
            if (v < min_s) min_s = v;
        }
    }
    
    int range = max_s - min_s;
    int scale = 255 / range;
    if (range > 0) {
        for (auto y = 0; y < channels[1].rows; y++) {
            for (auto x = 0; x < channels[1].cols; x++) {
                uchar v = channels[1].at<uchar>(y, x);
                channels[1].at<uchar>(y, x) = (v - min_s) * scale;
            }
        }
    }
    
    std::vector<cv::Mat> array_to_merge;
    array_to_merge.push_back(channels[0]);
    array_to_merge.push_back(channels[1]);
    array_to_merge.push_back(channels[2]);
    
    cv::Mat result_hsv;
    cv::merge(array_to_merge, result_hsv);
    
    cv::Mat result_bgr;
    cv::cvtColor(result_hsv, result_bgr, cv::COLOR_HSV2BGR);
    return result_bgr;
}



-(void) doGrabcutWithMaskImage:(UIImage*)image{
    [self showLoadingIndicatorView];

    __weak typeof(self)weakSelf = self;
    
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT,
                                             (unsigned long)NULL), ^(void) {
        UIImage* resultImage= [weakSelf.grabcut doGrabCutWithMask:weakSelf.resizedImage maskImage:[weakSelf resizeImage:image size:weakSelf.resizedImage.size] iterationCount:2];
        resultImage = [weakSelf masking:weakSelf.originalImage mask:[weakSelf resizeImage:resultImage size:weakSelf.originalImage.size]];
        dispatch_async(dispatch_get_main_queue(), ^(void) {
            [weakSelf.resultImageView setImage:resultImage];
            [weakSelf.imageView setAlpha:0.2];
            [weakSelf hideLoadingIndicatorView];
        });
    });
}

-(void) touchesBegan:(NSSet *)touches withEvent:(UIEvent *)event{
    NSLog(@"began");
    UITouch *touch = [touches anyObject];
    self.startPoint = [touch locationInView:self.imageView];
    
    if(_touchState == TouchStateNone || _touchState == TouchStateRect){
        [self.touchDrawView clear];
    }else if(_touchState == TouchStatePlus || _touchState == TouchStateMinus){
        [self.touchDrawView touchStarted:self.startPoint];
    }
}

-(void) touchesMoved:(NSSet *)touches withEvent:(UIEvent *)event{
    NSLog(@"moved");
    UITouch *touch = [touches anyObject];
    CGPoint point = [touch locationInView:self.imageView];
    
    if(_touchState == TouchStateRect){
        CGRect rect = [self getTouchedRect:_startPoint endPoint:point];
        [self.touchDrawView drawRectangle:rect];
    }else if(_touchState == TouchStatePlus || _touchState == TouchStateMinus){
        [self.touchDrawView touchMoved:point];
    }
}

-(void) touchesEnded:(NSSet *)touches withEvent:(UIEvent *)event{
    NSLog(@"ended");
    UITouch *touch = [touches anyObject];
    self.endPoint = [touch locationInView:self.imageView];
    
    if(_touchState == TouchStateRect){
        _grabRect = [self getTouchedRectWithImageSize:_resizedImage.size];
    }else if(_touchState == TouchStatePlus || _touchState == TouchStateMinus){
        [self.touchDrawView touchEnded:self.endPoint];
        _doGrabcutButton.enabled = YES;
    }
}


- (IBAction)tapOnReset:(id)sender {
    [self.imageView setImage:_originalImage];
    [self.resultImageView setImage:nil];
    [self.imageView setAlpha:1.0];
    _touchState = TouchStateNone;
    [self updateStateLabel];
    
    _rectButton.enabled = YES;
    _plusButton.enabled = NO;
    _minusButton.enabled = NO;
    _doGrabcutButton.enabled = NO;
    
    [self.touchDrawView clear];
    [self.grabcut resetManager];
}


- (IBAction)tapOnPlus:(id)sender {
    _touchState = TouchStatePlus;
    [self updateStateLabel];
    
    [_touchDrawView setCurrentState:TouchStatePlus];
}

- (IBAction)tapOnMinus:(id)sender {
    _touchState = TouchStateMinus;
    [self updateStateLabel];
    [_touchDrawView setCurrentState:TouchStateMinus];
}

-(IBAction)tapOnDoGrabcut:(id)sender{
    if(_touchState == TouchStateRect){
        if([self isUnderMinimumRect]){
            UIAlertView* alert = [[UIAlertView alloc] initWithTitle:@"Opps" message:@"More bigger rect for operation" delegate:nil cancelButtonTitle:nil otherButtonTitles:@"OK"  , nil];
            [alert show];
            
            return;
        }
        
        [self doGrabcut];
        [self.touchDrawView clear];
        
        _rectButton.enabled = NO;
        _plusButton.enabled = YES;
        _minusButton.enabled = YES;
        _doGrabcutButton.enabled = NO;
    }else if(_touchState == TouchStatePlus || _touchState == TouchStateMinus){
        UIImage* touchedMask = [self.touchDrawView maskImageWithPainting];
        [self doGrabcutWithMaskImage:touchedMask];
        
        [self.touchDrawView clear];
        _rectButton.enabled = NO;
        _plusButton.enabled = YES;
        _minusButton.enabled = YES;
        _doGrabcutButton.enabled = YES;
    }
}

-(BOOL) isUnderMinimumRect{
    if(_grabRect.size.width <20.0 || _grabRect.size.height < 20.0){
        return YES;
    }
    
    return NO;
}

#pragma mark - Image Picker

-(IBAction) tapOnPhoto:(id)sender{
    [self startMediaBrowserFromViewController: self
                                usingDelegate: self];
}

-(IBAction) tapOnCamera:(id)sender{
    [self startCameraControllerFromViewController: self
                                    usingDelegate: self];

}

-(void) setImageToTarget:(UIImage*)image{
    _originalImage = [self resizeWithRotation:image size:image.size];
    _resizedImage = [self getProperResizedImage:_originalImage];
    _imageView.image = _originalImage;
    [self.imageView setNeedsDisplay];
    [self initStates];
    [self.grabcut resetManager];
}

- (BOOL) startCameraControllerFromViewController: (UIViewController*) controller
                                   usingDelegate: (id <UIImagePickerControllerDelegate,
                                                   UINavigationControllerDelegate>) delegate {
    
    if (([UIImagePickerController isSourceTypeAvailable:
          UIImagePickerControllerSourceTypeCamera] == NO)
        || (delegate == nil)
        || (controller == nil))
        return NO;
    
    
    self.imagePicker = [[UIImagePickerController alloc] init];
    self.imagePicker.sourceType = UIImagePickerControllerSourceTypeCamera;
    
    // Displays a control that allows the user to choose picture or
    // movie capture, if both are available:
//    self.imagePicker.mediaTypes = [UIImagePickerController availableMediaTypesForSourceType:UIImagePickerControllerSourceTypeCamera];
    self.imagePicker.mediaTypes = @[(NSString *)kUTTypeImage];

    
    // Hides the controls for moving & scaling pictures, or for
    // trimming movies. To instead show the controls, use YES.
    self.imagePicker.allowsEditing = NO;
    
    self.imagePicker.delegate = delegate;
    
    [controller presentViewController:self.imagePicker animated:YES completion:nil];
    return YES;
}

- (BOOL) startMediaBrowserFromViewController: (UIViewController*) controller
                               usingDelegate: (id <UIImagePickerControllerDelegate,
                                               UINavigationControllerDelegate>) delegate {
    
    if (([UIImagePickerController isSourceTypeAvailable:
          UIImagePickerControllerSourceTypeSavedPhotosAlbum] == NO)
        || (delegate == nil)
        || (controller == nil))
        return NO;
    
    self.imagePicker = [[UIImagePickerController alloc] init];
    self.imagePicker.sourceType = UIImagePickerControllerSourceTypeSavedPhotosAlbum;
    
    // Displays saved pictures and movies, if both are available, from the
    // Camera Roll album.
    self.imagePicker.mediaTypes = @[(NSString *)kUTTypeImage];
//    [UIImagePickerController availableMediaTypesForSourceType:
//     UIImagePickerControllerSourceTypeSavedPhotosAlbum];
    
    // Hides the controls for moving & scaling pictures, or for
    // trimming movies. To instead show the controls, use YES.
    self.imagePicker.allowsEditing = NO;
    
    self.imagePicker.delegate = delegate;
    
    [controller presentViewController:self.imagePicker animated:YES completion:nil];
    return YES;
}

// For responding to the user tapping Cancel.
- (void) imagePickerControllerDidCancel: (UIImagePickerController *) picker {
    
    [self.imagePicker dismissViewControllerAnimated:YES completion:nil];
    self.imagePicker =nil;
}

// For responding to the user accepting a newly-captured picture or movie
- (void) imagePickerController: (UIImagePickerController *) picker
 didFinishPickingMediaWithInfo: (NSDictionary *) info {
    
    NSString *mediaType = [info objectForKey: UIImagePickerControllerMediaType];
    UIImage *originalImage, *editedImage, *resultImage;
    
    if (CFStringCompare ((CFStringRef) mediaType, kUTTypeImage, 0)
        == kCFCompareEqualTo) {
        
        editedImage = (UIImage *) [info objectForKey:
                                   UIImagePickerControllerEditedImage];
        originalImage = (UIImage *) [info objectForKey:
                                     UIImagePickerControllerOriginalImage];
        
        if (editedImage) {
            resultImage = editedImage;
        } else {
            resultImage = originalImage;
        }
    }
    
    [self setImageToTarget:resultImage];
    
    [self.imagePicker dismissViewControllerAnimated:YES completion:nil];
    self.imagePicker = nil;
    
    [self.imageView setImage:_originalImage];
    [self.resultImageView setImage:nil];
    [self.imageView setAlpha:1.0];
    _touchState = TouchStateNone;
    [self updateStateLabel];
    
    _rectButton.enabled = YES;
    _plusButton.enabled = NO;
    _minusButton.enabled = NO;
    _doGrabcutButton.enabled = NO;
    
    [self.touchDrawView clear];
    [self.grabcut resetManager];
    
}

#pragma mark - Indicator

CG_INLINE CGRect
CGRectSetOrigin(CGRect rect, CGPoint origin)
{
    rect.origin = origin;
    return rect;
}

- (void)showLoadingIndicatorView
{
    [self showLoadingIndicatorViewWithStyle:UIActivityIndicatorViewStyleWhite];
}

- (void)showLoadingIndicatorViewWithStyle:(UIActivityIndicatorViewStyle)activityIndicatorViewStyle
{
    if (self.spinner != nil) {
        [self hideLoadingIndicatorView];
    }
    
    self.dimmedView = [[UIView alloc] initWithFrame:CGRectMake(0, 0, self.view.frame.size.width, self.view.frame.size.height)];
    [self.dimmedView setBackgroundColor:[UIColor colorWithRed:0 green:0 blue:0 alpha:0.7]];
    [self.view addSubview:self.dimmedView];
    
    UIActivityIndicatorView *spinner = [[UIActivityIndicatorView alloc] initWithActivityIndicatorStyle:activityIndicatorViewStyle];
    spinner.frame = CGRectSetOrigin(spinner.frame, CGPointMake(floorf(CGRectGetMidX(self.view.bounds) - CGRectGetMidX(spinner.bounds)), floorf(CGRectGetMidY(self.view.bounds) - CGRectGetMidY(spinner.bounds))));
    spinner.autoresizingMask = UIViewAutoresizingFlexibleLeftMargin|UIViewAutoresizingFlexibleRightMargin|UIViewAutoresizingFlexibleTopMargin|UIViewAutoresizingFlexibleBottomMargin;
    [spinner startAnimating];
    [self.view addSubview:spinner];
    self.spinner = spinner;
    
    [self.view setUserInteractionEnabled:NO];
}

- (void)hideLoadingIndicatorView
{
    [self.spinner stopAnimating];
    [self.spinner removeFromSuperview];
    self.spinner = nil;
    
    [self.dimmedView removeFromSuperview];
    self.dimmedView = nil;
    
    [self.view setUserInteractionEnabled:YES];
}

- (IBAction)saveToPhotoRollAction:(id)sender {
   
    UIImageWriteToSavedPhotosAlbum( self.resultImageView.image, nil, nil, nil);
}


@end

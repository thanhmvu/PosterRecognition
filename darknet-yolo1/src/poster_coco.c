#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "string.h"

#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>


#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif


// char *poster_names[] = {"class0", "class1", "class2", "class3", "class4", "class5", "class6", "class7", "class8", "class9", "class10", "class11", "class12", "class13", "class14", "class15", "class16", "class17", "class18", "class19", "class20", "class21", "class22", "class23", "class24", "class25", "class26", "class27", "class28", "class29"};
// image poster_labels[30];
char *poster_names[100];

// input for multi_validate
int cfg_tipp = 2000; //train images per poster
int cfg_classes = 20;
int trial_idx = 2;
int start_weight = 5;
int num_weight = 93;
/*==================================== Helper methods =====================================*/

void convert_poster_detections(float *predictions, int classes, int num, int square, int side, int w, int h, float thresh, float **probs, box *boxes, int only_objectness)
{
    int i,j,n;
    for (i = 0; i < side*side; ++i){
        int row = i / side;
        int col = i % side;
        for(n = 0; n < num; ++n){
            int index = i*num + n;
            int p_index = side*side*classes + i*num + n;
            float scale = predictions[p_index];
            int box_index = side*side*(classes + num) + (i*num + n)*4;
            boxes[index].x = (predictions[box_index + 0] + col) / side * w;
            boxes[index].y = (predictions[box_index + 1] + row) / side * h;
            boxes[index].w = pow(predictions[box_index + 2], (square?2:1)) * w;
            boxes[index].h = pow(predictions[box_index + 3], (square?2:1)) * h;
            for(j = 0; j < classes; ++j){
                int class_index = i*classes;
                float prob = scale*predictions[class_index+j];
                probs[index][j] = (prob > thresh) ? prob : 0;
            }
            if(only_objectness){
                probs[index][0] = scale;
            }
        }
    }
}

int updateCorrect(int num, float thresh, float **probs, int classes, char * path, int correct, int *arr){
		float max_prob = 0; float max_prob2 = 0;
		int max_class = -1; int max_class2 = -1;
		int j;
		for(j = 0; j < num; ++j){
				int class = max_index(probs[j], classes);
				float prob = probs[j][class];
				if(prob > thresh){
						if (prob> max_prob){
								max_prob = prob;
								max_class = class;
						} else if (prob> max_prob2) {
								max_prob2 = prob;
								max_class2 = class; } } }
	
		// extract the class from the path, provided that the path's format is path/to/xxxxxx_yyyyyy.jpg, where xxxxxx is the class index.
		int truthClass = get_poster_class(path); // in utils.c
	
		// Print and update
		printf("==] THANH: image %s\n", path);
		printf("1st poster: %d - %.0f%%,    2nd poster: %d - %.0f%%\n",  max_class, max_prob*100,max_class2, max_prob2*100);
		arr[0] = max_class;
		arr[1] = max_prob*100;
		arr[2] = max_class2;
		arr[3] = max_prob2*100;
		
		if (max_class == truthClass){ correct++; }
		else{printf("====================================================== WRONG \n");}
		return correct;
}

char * getFolder(char * path){
	char d = 0;
	char i;
	for(i=0; path[i]!='\0'; ++i){
		if(path[i]=='/'){ d = i; }
	}
	int length = d + 1;
	char * folder = malloc(length);
	memcpy(folder, path, length);
	folder[length] = '\0';
	return folder;
}

/*==================================== Main methods =====================================*/

void train_poster(char *cfgfile, char *weightfile, char *train_images, char *backup_directory)
{
		
		// create trainData.txt
		char dataFile[255];	
		char t[100];
		time_t now = time(0);
		strftime (t, 100, "%Y%m%d-%H%M%S", localtime (&now));
		sprintf(dataFile,"%s/trainData_%s.txt",getFolder(cfgfile),t);
		FILE * file = fopen(dataFile, "w+");
	
    srand(time(0));
    data_seed = time(0);
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
		fprintf(file, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = net.batch*net.subdivisions;
    int i = *net.seen/imgs;
    data train, buffer;


    layer l = net.layers[net.n - 1];

    int side = l.side;
    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = side;
    args.d = &buffer;
    args.type = REGION_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net.max_batches){
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_in_thread(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
        float loss = train_network(net, train);
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
				fprintf(file, "%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
        if(i%1000==0 || (i < 1000 && i%100 == 0)){
						fflush(file);
				}
        if(i%1000==0 || (i < 1000 && i%100 == 0)){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
		fclose(file);
}

void validate_poster(char *cfgfile, char *weightfile, char * testImgs, int savingImg)
{
		char results[255];
		if(savingImg != 0){
			// create folder "results/" in test folder to store output images
			sprintf(results,"%sresults/",getFolder(cfgfile));
			mkdir(results,0777);
		}
		
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

		char * path = testImgs;
		list *plist = get_paths(path);
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int square = l.sqrt;
    int side = l.side;

    int j;
    box *boxes = calloc(side*side*l.n, sizeof(box));
    float **probs = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;
    int t;

    float thresh = 0.0;
		// don't use nms since we only "classifying" posters, not detecting them"
//     int nms = 1;
//     float iou_thresh = .5;

    int nthreads = 2;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.type = IMAGE_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
	
    int total = 0;
    int correct = 0;
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            float *predictions = network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
//             convert_poster_detections(predictions, classes, l.n, square, side, w, h, thresh, probs, boxes, 0);
						// Fixed a bug here by changing w and h to 1, 1 (similar to the test_poster method)
						convert_poster_detections(predictions, classes, l.n, square, side, 1, 1, thresh, probs, boxes, 0);
//             if (nms) do_nms_sort(boxes, probs, side*side*l.n, classes, iou_thresh);

						if(savingImg == 0){
							int arr[4]; // array to store [best_match, best_prob, 2ndB_match, 2ndB_prob]
							// Calculate the accuracy of classification using hardcored format of the path
            	correct = updateCorrect(side *side *l.n, thresh, probs, classes,path, correct, arr);
						} 
						else {
							char * imgName = get_file_name(path);
							char * classIdx = strtok(strdup(imgName),"_"); // split a copy of "imgName"
							
							// create folder "results/classIdx/" to store images of same class
							char classFolder[256]; // don't use char * to avoid segmentation fault
							sprintf(classFolder,"%s%s/",results,classIdx);
							mkdir(classFolder,0777);
							
							int lastCorrect = correct;
							// Calculate the accuracy of classification using hardcored format of the path
							int arr[4]; // array to store [best_match, best_prob, 2ndB_match, 2ndB_prob]
							correct = updateCorrect(side *side *l.n, thresh, probs, classes,path, correct, arr);
							if (correct > lastCorrect){ // if correct, assign folder "1"
								sprintf(classFolder,"%s%s/",classFolder,"1");
							} else { // if not, assign folder "0"
								sprintf(classFolder,"%s%s/",classFolder,"0");
							}
							mkdir(classFolder,0777);
							image im = load_image_color(path,0,0);
// 							draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, poster_names, poster_labels, classes);
							// fixed a crash here by removing poster_labels
							draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, poster_names, 0, classes);
								
							char imgToSave[256];
							sprintf(imgToSave, "%s%s_%d_%d_%d_%d.jpg", classFolder, get_image_name(path), 
											arr[0], arr[1], arr[2], arr[3]);
							
							int h = 500;
							im = resize_image(im,h*im.w/im.h,h);
							save_image(im, imgToSave);
						}
            total++;
            printf("Current detection accuracy: %.02f%%\n", correct*100.0/total);	

            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    printf("Final detection accuracy: %.02f%%\n", correct*100.0/total);
    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
}


/*
void multi_validate()
{
	
	char *cfgfile= (char*)malloc(255 * sizeof(char));
	char *testImgs= (char*)malloc(255 * sizeof(char));
	char *weightTemplate= (char*)malloc(255 * sizeof(char));
	sprintf (cfgfile, "../../database/realworld/set2/randTrain/%dC_%dP_trial%d/poster_detect_%dc.cfg", cfg_classes, cfg_tipp, trial_idx, cfg_classes);
	sprintf (testImgs, "../../database/realworld/set2/randTest/%dC_trial%d/test.txt", cfg_classes, trial_idx);
	sprintf (weightTemplate, "../../database/realworld/set2/randTrain/%dC_%dP_trial%d/backup/detect_weights/poster_detect_%dc_%s.weights", cfg_classes,cfg_tipp,trial_idx,cfg_classes,"\%d");
	
	
// 	char * cfgfile = "../../database/realworld/set2/randTrain/%dC_%dP_trial%d/poster_detect_%dc.cfg" % (cfg_classes,cfg_tipp,trial_idx,cfg_classes);
// 	char * testImgs = "../../database/realworld/set2/randTest/%dC_trial%d/test.txt" % (cfg_classes, trial_idx);
// 	char * weightTemplate = "../../database/realworld/set2/randTrain/%dC_%dP_trial%d/backup/detect_weights/poster_detect_%dc_%d.weights" % (cfg_classes,cfg_tipp,trial_idx,cfg_classes,1);
// // 	int testWeights[] = {17000,18000,19000,20000,21000,22000};
	
	// whether to save images as visualization or not
	int savingImg = 0;
	
	char dataFile[255];	
	char t[100];
	time_t now = time(0);
	strftime (t, 100, "%Y%m%d-%H%M%S", localtime (&now));
	sprintf(dataFile,"%s/testData_%s.txt",getFolder(testImgs),t);
	FILE * file = fopen(dataFile, "w+");
	
	fprintf(file, cfgfile); fprintf(file, "\n");
	fprintf(file, testImgs); fprintf(file, "\n");

  network net = parse_network_cfg(cfgfile);
	
	int weight_idx;
// 	for(i=0; i< sizeof(testWeights)/sizeof(testWeights[0]); i++){
	// loop and test each weights
	for(weight_idx=0; weight_idx< num_weight; weight_idx++){
		int testWeight = (start_weight+weight_idx)*1000;//testWeights[i]
		char weightfile[255];
		sprintf(weightfile,weightTemplate,testWeight);
		fprintf(file, weightfile); fprintf(file, "\n");
		
		char results[255];
		if(savingImg != 0){
			// create folder "results/" in test folder to store output images
			sprintf(results,"%sresults/",getFolder(cfgfile));
			mkdir(results,0777);
		}
		
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

		char * path = testImgs;
		list *plist = get_paths(path);
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int square = l.sqrt;
    int side = l.side;

    int j;
    box *boxes = calloc(side*side*l.n, sizeof(box));
    float **probs = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;
    int t;

    float thresh = 0.0;
		// don't use nms since we only "classifying" posters, not detecting them"
//     int nms = 1;
//     float iou_thresh = .5;

    int nthreads = 2;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.type = IMAGE_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
	
    int total = 0;
    int correct = 0;
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            float *predictions = network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
//             convert_poster_detections(predictions, classes, l.n, square, side, w, h, thresh, probs, boxes, 0);
						// Fixed a bug here by changing w and h to 1, 1 (similar to the test_poster method)
						convert_poster_detections(predictions, classes, l.n, square, side, 1, 1, thresh, probs, boxes, 0);
//             if (nms) do_nms_sort(boxes, probs, side*side*l.n, classes, iou_thresh);

						if(savingImg == 0){
							int arr[4]; // array to store [best_match, best_prob, 2ndB_match, 2ndB_prob]
							// Calculate the accuracy of classification using hardcored format of the path
            	correct = updateCorrect(side *side *l.n, thresh, probs, classes,path, correct, arr);
						} 
						else {
							char * imgName = get_file_name(path);
							char * classIdx = strtok(strdup(imgName),"_"); // split a copy of "imgName"
							
							// create folder "results/classIdx/" to store images of same class
							char classFolder[256]; // don't use char * to avoid segmentation fault
							sprintf(classFolder,"%s%s/",results,classIdx);
							mkdir(classFolder,0777);
							
							int lastCorrect = correct;
							// Calculate the accuracy of classification using hardcored format of the path
							int arr[4]; // array to store [best_match, best_prob, 2ndB_match, 2ndB_prob]
							correct = updateCorrect(side *side *l.n, thresh, probs, classes,path, correct, arr);
							if (correct > lastCorrect){ // if correct, assign folder "1"
								sprintf(classFolder,"%s%s/",classFolder,"1");
							} else { // if not, assign folder "0"
								sprintf(classFolder,"%s%s/",classFolder,"0");
							}
							mkdir(classFolder,0777);
							image im = load_image_color(path,0,0);
// 							draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, poster_names, poster_labels, classes);
							// fixed a crash here by removing poster_labels
							draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, poster_names, 0, classes);
								
							char imgToSave[256];
							sprintf(imgToSave, "%s%s_%d_%d_%d_%d.jpg", classFolder, get_image_name(path), 
											arr[0], arr[1], arr[2], arr[3]);
							
							int h = 500;
							im = resize_image(im,h*im.w/im.h,h);
							save_image(im, imgToSave);
						}
            total++;
            printf("Current detection accuracy: %.02f%%\n", correct*100.0/total);	

            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    printf("Final detection accuracy: %.02f%%\n", correct*100.0/total);
    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
		fprintf(file, "Detection accuracy: %.02f%%\n", correct*100.0/total);
		
		if(weight_idx%10==0){
			fflush(file);
		}
	}
	fclose(file);
}
*/


void multi_validate()
{
	
	char *cfgfile= "../../database/realworld/set2/train/50C_5000P_s0_train/poster_detect.cfg";
	char *testImgs= "../../database/realworld/set2/test/real_images/testtxt/test_50c.txt";
	char *weightTemplate= "../../database/realworld/set2/train/50C_5000P_s0_train/backup/detect_weights/poster_detect_%d.weights";
	
	// whether to save images as visualization or not
	int savingImg = 0;
	
	char dataFile[255];	
	char t[100];
	time_t now = time(0);
	strftime (t, 100, "%Y%m%d-%H%M%S", localtime (&now));
	sprintf(dataFile,"%s/testData_%s.txt",getFolder(testImgs),t);
	FILE * file = fopen(dataFile, "w+");
	
	fprintf(file, cfgfile); fprintf(file, "\n");
	fprintf(file, testImgs); fprintf(file, "\n");

  network net = parse_network_cfg(cfgfile);
	
	int weight_idx;
	// loop and test each weights
	for(weight_idx = 155; weight_idx< 167; weight_idx ++){
		int testWeight = (weight_idx)*1000;//testWeights[i]
		char weightfile[255];
		sprintf(weightfile,weightTemplate,testWeight);
		fprintf(file, weightfile); fprintf(file, "\n");
		
		char results[255];
		if(savingImg != 0){
			// create folder "results/" in test folder to store output images
			sprintf(results,"%sresults/",getFolder(cfgfile));
			mkdir(results,0777);
		}
		
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

		char * path = testImgs;
		list *plist = get_paths(path);
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int square = l.sqrt;
    int side = l.side;

    int j;
    box *boxes = calloc(side*side*l.n, sizeof(box));
    float **probs = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;
    int t;

    float thresh = 0.0;
		// don't use nms since we only "classifying" posters, not detecting them"
//     int nms = 1;
//     float iou_thresh = .5;

    int nthreads = 2;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.type = IMAGE_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
	
    int total = 0;
    int correct = 0;
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            float *predictions = network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
//             convert_poster_detections(predictions, classes, l.n, square, side, w, h, thresh, probs, boxes, 0);
						// Fixed a bug here by changing w and h to 1, 1 (similar to the test_poster method)
						convert_poster_detections(predictions, classes, l.n, square, side, 1, 1, thresh, probs, boxes, 0);
//             if (nms) do_nms_sort(boxes, probs, side*side*l.n, classes, iou_thresh);

						if(savingImg == 0){
							int arr[4]; // array to store [best_match, best_prob, 2ndB_match, 2ndB_prob]
							// Calculate the accuracy of classification using hardcored format of the path
            	correct = updateCorrect(side *side *l.n, thresh, probs, classes,path, correct, arr);
						} 
						else {
							char * imgName = get_file_name(path);
							char * classIdx = strtok(strdup(imgName),"_"); // split a copy of "imgName"
							
							// create folder "results/classIdx/" to store images of same class
							char classFolder[256]; // don't use char * to avoid segmentation fault
							sprintf(classFolder,"%s%s/",results,classIdx);
							mkdir(classFolder,0777);
							
							int lastCorrect = correct;
							// Calculate the accuracy of classification using hardcored format of the path
							int arr[4]; // array to store [best_match, best_prob, 2ndB_match, 2ndB_prob]
							correct = updateCorrect(side *side *l.n, thresh, probs, classes,path, correct, arr);
							if (correct > lastCorrect){ // if correct, assign folder "1"
								sprintf(classFolder,"%s%s/",classFolder,"1");
							} else { // if not, assign folder "0"
								sprintf(classFolder,"%s%s/",classFolder,"0");
							}
							mkdir(classFolder,0777);
							image im = load_image_color(path,0,0);
// 							draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, poster_names, poster_labels, classes);
							// fixed a crash here by removing poster_labels
							draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, poster_names, 0, classes);
								
							char imgToSave[256];
							sprintf(imgToSave, "%s%s_%d_%d_%d_%d.jpg", classFolder, get_image_name(path), 
											arr[0], arr[1], arr[2], arr[3]);
							
							int h = 500;
							im = resize_image(im,h*im.w/im.h,h);
							save_image(im, imgToSave);
						}
            total++;
            printf("Current detection accuracy: %.02f%%\n", correct*100.0/total);	

            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    printf("Final detection accuracy: %.02f%%\n", correct*100.0/total);
    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
		fprintf(file, "Detection accuracy: %.02f%%\n", correct*100.0/total);
		
		if(weight_idx%25==0){
			fflush(file);
		}
	}
	fclose(file);
}


void test_poster(char *cfgfile, char *weightfile, char *filename, float thresh)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    detection_layer l = net.layers[net.n-1];
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.5;
    box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = resize_image(im, net.w, net.h);
        float *X = sized.data;
        time=clock();
        float *predictions = network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        convert_poster_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
      
        if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
//         draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, poster_names, poster_labels, l.classes);
        draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, poster_names, 0, l.classes);
				char out[256];
				char * outpath = "/home/vut/PosterRecognition/DeepNet/database/realworld/set2/test/real_images/%s";
				sprintf(out, outpath, get_file_name(filename));
        save_image(im, out);
//         show_image(im, "predictions");

//         show_image(sized, "resized");
        free_image(im);
        free_image(sized);
#ifdef OPENCV
//         cvWaitKey(0);
//         cvDestroyAllWindows();
#endif
        if (filename) break;
    }
}


void run_poster_detect(int argc, char **argv)
{
    float thresh = find_float_arg(argc, argv, "-thresh", .2);
    int savingImg = find_int_arg(argc, argv, "-saveImg", 0);
    if(argc < 4){
        fprintf(stderr, "usage: ./darknet poster_detect [train/test/valid] [cfg] [weights (optional)]\n");
        return;
    }

		char *cfg = argv[3];
		char *weights = (argc > 4) ? argv[4] : 0;
		char *filename = (argc > 5) ? argv[5]: 0;
    if(0==strcmp(argv[2], "train")){
			filename = (argc > 4) ? argv[4]: 0;
			char *backup = (argc > 5) ? argv[5]: 0;
			weights = (argc > 6) ? argv[6] : 0;
			train_poster(cfg, weights, filename, backup);
		}
    else if(0==strcmp(argv[2], "valid")) validate_poster(cfg, weights, filename, savingImg);
    else if(0==strcmp(argv[2], "test")) test_poster(cfg, weights, filename, thresh);
    else if(0==strcmp(argv[2], "multivalid")) multi_validate();
}

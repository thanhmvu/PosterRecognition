#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_list.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif
static int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};
char *poster_names[] = {"000000", "000001", "000002", "000003", "000004", "000005", "000006", "000007", "000008", "000009", "000010", "000011", "000012", "000013", "000014", "000015", "000016", "000017", "000018", "000019", "000020", "000021", "000022", "000023", "000024", "000025", "000026", "000027", "000028", "000029", "000030", "000031", "000032", "000033", "000034", "000035", "000036", "000037", "000038", "000039", "000040", "000041", "000042", "000043", "000044", "000045", "000046", "000047", "000048", "000049", "000050", "000051", "000052", "000053", "000054", "000055", "000056", "000057", "000058", "000059", "000060", "000061", "000062", "000063", "000064", "000065", "000066", "000067", "000068", "000069", "000070", "000071", "000072", "000073", "000074", "000075", "000076", "000077", "000078", "000079", "000080", "000081", "000082", "000083", "000084", "000085", "000086", "000087", "000088", "000089", "000090", "000091", "000092", "000093", "000094", "000095", "000096", "000097", "000098", "000099"};

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


int get_poster_class(char * path){
  // methods to extract the class from the path
	// provided that the path's format is <path-to-image>/xxxxxx_yyyyyy.jpg
	// where xxxxxx is the class index.
	char* copy = malloc (1 + strlen(path));
	strcpy(copy,path);
	char* dim = "/._"; // divide string by "/","." and "_"
	char* iterator = strtok(copy,dim);
  char* buff = "-1"; 
	char* class = "-1";
	while(strcmp(iterator,"jpg") != 0){
	    class = buff;
	    buff = iterator;
	    iterator = strtok(NULL,dim);
	}
// 	printf("Loaded %s, class %d\n",path,atoi(class));
  return atoi(class);
}

char * get_file_name(char * path){
	char* copy = malloc (1 + strlen(path));
	strcpy(copy,path);
	char* dim = "/"; // divide string by "/"
	char* iterator = strtok(copy,dim);
	char* name = "-1";
	while(iterator != NULL){
	    name = iterator;
	    iterator = strtok(NULL,dim);
	}
// 	printf("Loaded %s, class %d\n",path,atoi(class));
  return name;
}

char * get_image_name(char * path){
	char* copy = malloc (1 + strlen(path));
	strcpy(copy,path);
	char* dim = "/."; // divide string by "/","."
	char* iterator = strtok(copy,dim);
	char* name = "-1";
	while(strcmp(iterator,"jpg") != 0){
	    name = iterator;
	    iterator = strtok(NULL,dim);
	}
  return name;
}

char * get_second_last(char * path, char * dim){
	char* copy = malloc (1 + strlen(path));
	strcpy(copy,path);
	char* iterator = strtok(copy,dim);
	char* name = "-1";
  char* buff = "-1"; 
	while(iterator != NULL){
	    name = buff;
	    buff = iterator;
	    iterator = strtok(NULL,dim);
	}
// 	printf("Loaded %s, class %d\n",path,atoi(class));
  return name;
}

/*=======================================================================================*/
/*===                                                                                 ===*/
/*===                                  Main methods                                   ===*/
/*===                                                                                 ===*/
/*=======================================================================================*/

// void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
void train_detector(int *gpus, int ngpus, int clear)
{
		/*
		To train the net, need:
			+ cfgfile
			+ train_images path (train.txt)
			+ backup path
			+ weights
		*/
		
		int cfg_classes = 100;
// 		char * note = "noblur";
		char * note = "trial3";
		int cfg_imgs_per_class = 2000; //train images per poster
		
		char *cfg_train_dir = (char*)malloc(255 * sizeof(char));
		char *cfg_train_dir_format = "/home/vut/PosterRecognition/DeepNet/database/realworld/set2/randTrain/%dC_%dP_%s/"; 
		sprintf(cfg_train_dir, cfg_train_dir_format, cfg_classes, cfg_imgs_per_class, note); 
	
		char *cfgfile = (char*)malloc(255 * sizeof(char));
		sprintf(cfgfile, "/home/vut/PosterRecognition/DeepNet/database/realworld/set2/randTrain/cfg-yolo2/yolo2_%dc.cfg", cfg_classes); 
    
		char *train_images = (char*)malloc(255 * sizeof(char));
		sprintf(train_images, "%srandTrain.txt", cfg_train_dir);	
	
		char *backup_directory = (char*)malloc(255 * sizeof(char));
		sprintf(backup_directory, "%sbackup/yolo2_weights/", cfg_train_dir);	
		mkdir(backup_directory, 0700);
	
		char *weightfile = "/home/vut/PosterRecognition/DeepNet/database/darknet19_448.conv.23";
// 		char *weightfile = "/home/vut/PosterRecognition/DeepNet/database/realworld/set2/randTrain/20C_2000P_trial1/backup/yolo2_weights/poster_15000.weights";
	
//     list *options = read_data_cfg(datacfg);
//     char *train_images = option_find_str(options, "train", "data/train.list");
//     char *backup_directory = option_find_str(options, "backup", "/backup/");
	
    // create file to store train data
    char dataFile[255];	
    char t[100];
    time_t now = time(0);
    strftime (t, 100, "%Y%m%d-%H%M%S", localtime (&now));
    sprintf(dataFile,"%strainYolo2_%s.txt", cfg_train_dir, t);
    FILE * file = fopen(dataFile, "w+");
	
		fprintf(file, "Number of classes: %d, Note: %s, Number of images per class: %d\n", cfg_classes, note, cfg_imgs_per_class);
		fprintf(file, "Config file: \n%s\n",cfgfile);
		fprintf(file, "Train images: \n%s\n",train_images);
		fprintf(file, "Backup dir: \n%s\n",backup_directory);
		fprintf(file, "Input weights: \n%s\n\n",weightfile);
	
    printf("\nNumber of classes: %d, Note: %s, Number of images per class: %d\n", cfg_classes, note, cfg_imgs_per_class);
		printf("Config file: \n%s\n",cfgfile);
		printf("Train images: \n%s\n",train_images);
		printf("Backup dir: \n%s\n",backup_directory);
		printf("Input weights: \n%s\n\n",weightfile);
	
	
    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network *nets = calloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = parse_network_cfg(cfgfile);
        if(weightfile){
            load_weights(&nets[i], weightfile);
        }
        if(clear) *nets[i].seen = 0;
        nets[i].learning_rate *= ngpus;
    }
    srand(time(0));
    network net = nets[0];

    int imgs = net.batch * net.subdivisions * ngpus;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
	fprintf(file, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    data train, buffer;

    layer l = net.layers[net.n - 1];

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
    args.num_boxes = l.max_boxes;
    args.d = &buffer;
    args.type = DETECTION_DATA;
    args.threads = 8;

    args.angle = net.angle;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;

    pthread_t load_thread = load_data(args);
    clock_t time;
    int count = 0;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net.max_batches){
        if(l.random && count++%10 == 0){
            printf("Resizing\n");
            int dim = (rand() % 10 + 10) * 32;
            if (get_current_batch(net)+100 > net.max_batches) dim = 544;
            //int dim = (rand() % 4 + 16) * 32;
            printf("%d\n", dim);
            args.w = dim;
            args.h = dim;

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            for(i = 0; i < ngpus; ++i){
                resize_network(nets + i, dim, dim);
            }
            net = nets[0];
        }
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        /*
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[10] + 1 + k*5);
           if(!b.x) break;
           printf("loaded: %f %f %f %f\n", b.x, b.y, b.w, b.h);
           }
           image im = float_to_image(448, 448, 3, train.X.vals[10]);
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[10] + 1 + k*5);
           printf("%d %d %d %d\n", truth.x, truth.y, truth.w, truth.h);
           draw_bbox(im, b, 8, 1,0,0);
           }
           save_image(im, "truth11");
         */

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        i = get_current_batch(net);
        printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
	    fprintf(file, "%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
        if(i%1000==0 || (i < 1000 && i%100 == 0)){
			fflush(file);
		}
        if(i%1000==0 || (i < 1000 && i%100 == 0)){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
#ifdef GPU
    if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
    fclose(file);
}



void multivalidate_detector()
{
		int cfg_classes = 100;
//  		int cfg_trial_idx = 1;
		char * note = "trial1";
		int cfg_imgs_per_class = 2000; //train images per poster
		int start_weight = 80;
		int end_weight = 80; 
		
		// whether to save images as visualization or not
		int savingImg = 1;
	
		char *cfgfile= (char*)malloc(255 * sizeof(char));
		sprintf (cfgfile, "/home/vut/PosterRecognition/DeepNet/database/realworld/set2/randTrain/cfg-yolo2/yolo2_%dc.cfg", cfg_classes);
	
		char *valid_images= (char*)malloc(255 * sizeof(char));
		sprintf (valid_images, "/home/vut/PosterRecognition/DeepNet/database/realworld/set2/randTest/%dC_%dP_%s/test.txt", cfg_classes, cfg_imgs_per_class, note);

		char *weightTemplate= (char*)malloc(255 * sizeof(char));
		sprintf (weightTemplate, "/home/vut/PosterRecognition/DeepNet/database/realworld/set2/randTrain/%dC_%dP_%s/backup/yolo2_weights/yolo2_%dc_%s.weights", cfg_classes,cfg_imgs_per_class,note,cfg_classes,"\%d");
// 		sprintf (weightTemplate, "/home/vut/PosterRecognition/DeepNet/database/realworld/set2/randTrain/%dC_%dP_trial%d/backup/yolo2_weights/poster_%%d.weights", cfg_classes, cfg_imgs_per_class, cfg_trial_idx);
    
		char dataFile[255];	
		char t[100];
		time_t now = time(0);
		strftime (t, 100, "%Y%m%d-%H%M%S", localtime (&now));
		sprintf(dataFile,"%s/testYolo2_%s.txt",getFolder(valid_images),t);
		FILE * file = fopen(dataFile, "w+");
	
		fprintf(file, "=> Output test file: \n%s\n", dataFile);
		fprintf(file, "=> Number of classes: %d \n=> Note: %s, \n=> Number of images per class: %d \n=> Weights: %dk to %dk \n\n", cfg_classes, note, cfg_imgs_per_class, start_weight, end_weight);
		fprintf(file, "=> Config file: \n%s\n", cfgfile);
		fprintf(file, "=> Validated images: \n%s\n", valid_images);
		fprintf(file, "=> Input weight template: \n%s\n\n",weightTemplate);
	
		printf("=> Output test file: \n%s\n", dataFile);
		printf("=> Number of classes: %d \n=> Note: %s, \n=> Number of images per class: %d \n=> Weights: %dk to %dk \n\n", cfg_classes, note, cfg_imgs_per_class, start_weight, end_weight);
		printf("=> Config file:\n %s\n", cfgfile);
		printf("=> Validated images: \n%s\n", valid_images);
		printf("=> Input weight template: \n%s\n\n",weightTemplate);
    
//     int j;
//     list *options = read_data_cfg(datacfg);
//     char *valid_images = option_find_str(options, "valid", "data/train.list");
//     char *name_list = option_find_str(options, "names", "data/names.list");
//     char *prefix = option_find_str(options, "results", "results");
//     char **names = get_labels(name_list);
//     char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
//     if (mapf) map = read_map(mapf);

    network net = parse_network_cfg(cfgfile);
    
		int weight_idx;
		for(weight_idx=start_weight; weight_idx<=end_weight; weight_idx++){
				int total = 0;
				int correct = 0;
        
				int testWeight = weight_idx*1000;//testWeights[i]
				char weightfile[255];
				sprintf(weightfile,weightTemplate,testWeight);
				fprintf(file, weightfile); fprintf(file, "\n");
        
				char results[255];
				if(savingImg != 0){
						// create folder to store output images
						sprintf(results,"%sresults/",getFolder(valid_images));
						mkdir(results,0777);
				}
        
        if(weightfile){
            load_weights(&net, weightfile);
        }
        set_batch_network(&net, 1);
        fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
        srand(time(0));

        char *base = "comp4_det_test_";
        list *plist = get_paths(valid_images);
        char **paths = (char **)list_to_array(plist);

        layer l = net.layers[net.n-1];
        int classes = l.classes;

    //     char buff[1024];
    //     char *type = option_find_str(options, "eval", "voc");
    //     FILE *fp = 0;
    //     FILE **fps = 0;
    //     int coco = 0;
    //     int imagenet = 0;
    //     if(0==strcmp(type, "coco")){
    //         snprintf(buff, 1024, "%s/coco_results.json", prefix);
    //         fp = fopen(buff, "w");
    //         fprintf(fp, "[\n");
    //         coco = 1;
    //     } else if(0==strcmp(type, "imagenet")){
    //         snprintf(buff, 1024, "%s/imagenet-detection.txt", prefix);
    //         fp = fopen(buff, "w");
    //         imagenet = 1;
    //         classes = 200;
    //     } else {
    //         fps = calloc(classes, sizeof(FILE *));
    //         for(j = 0; j < classes; ++j){
    //             snprintf(buff, 1024, "%s/%s%s.txt", prefix, base, names[j]);
    //             fps[j] = fopen(buff, "w");
    //         }
    //     }

        int j;
        box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
        float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
        for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

        int m = plist->size;
        int i=0;
        int t;

        float thresh = .005;
        float nms = .45;

        int nthreads = 4;
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
                network_predict(net, X);
                int w = val[t].w;
                int h = val[t].h;
                get_region_boxes(l, w, h, thresh, probs, boxes, 0, map);
                if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, classes, nms);
               
//                 if (coco){
//                     print_cocos(fp, path, boxes, probs, l.w*l.h*l.n, classes, w, h);
//                 } else if (imagenet){
//                     print_imagenet_detections(fp, i+t-nthreads+1, boxes, probs, l.w*l.h*l.n, classes, w, h);
//                 } else {
//                     print_detector_detections(fps, id, boxes, probs, l.w*l.h*l.n, classes, w, h);
//                 }
                
								if(savingImg == 0){
								int arr[4]; // array to store [best_match, best_prob, 2ndB_match, 2ndB_prob]
								// Calculate the accuracy of classification using hardcored format of the path
								correct = updateCorrect(l.w *l.h *l.n, thresh, probs, classes, path, correct, arr);
								}else {
										char * imgName = get_file_name(path);
										char * classIdx = strtok(strdup(imgName),"_"); // split a copy of "imgName"

										// create folder "results/classIdx/" to store images of same class
										char classFolder[256]; // don't use char * to avoid segmentation fault
										sprintf(classFolder,"%s%s/",results,classIdx);
										mkdir(classFolder,0777);

										int lastCorrect = correct;
										// Calculate the classification accuracy using HARDCODED path format
										// array to store [best_match, best_prob, 2ndB_match, 2ndB_prob]
										int arr[4]; 
										correct = updateCorrect(l.w *l.h *l.n, thresh, probs, classes,path, correct, arr);
										if (correct > lastCorrect){ // if correct, assign folder "1"
												sprintf(classFolder,"%s%s/",classFolder,"1");
										} else { // if not, assign folder "0"
												sprintf(classFolder,"%s%s/",classFolder,"0");
										}
										mkdir(classFolder,0777);
										image im = load_image_color(path,0,0);
										// fixed a crash here by removing poster_labels
										// draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, poster_names, poster_labels, classes);
										// use thresh = .1 to reduce boxes drawn
										draw_detections(im, l.w*l.h*l.n, .1, boxes, probs, poster_names, 0, classes);

										char imgToSave[256];
										sprintf(imgToSave, "%s%s_%d_%d_%d_%d.jpg", classFolder, get_image_name(path), arr[0], arr[1], arr[2], arr[3]);

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

				if(weight_idx%5==0){
						fflush(file);
				}
	//         for(j = 0; j < classes; ++j){
	//             if(fps) fclose(fps[j]);
	//         }
	//         if(coco){
	//             fseek(fp, -2, SEEK_CUR); 
	//             fprintf(fp, "\n]\n");
	//             fclose(fp);
	//         }
	//         fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
		}
    fclose(file);
}



static int get_coco_image_id(char *filename)
{
    char *p = strrchr(filename, '_');
    return atoi(p+1);
}

static void print_cocos(FILE *fp, char *image_path, box *boxes, float **probs, int num_boxes, int classes, int w, int h)
{
    int i, j;
    int image_id = get_coco_image_id(image_path);
    for(i = 0; i < num_boxes; ++i){
        float xmin = boxes[i].x - boxes[i].w/2.;
        float xmax = boxes[i].x + boxes[i].w/2.;
        float ymin = boxes[i].y - boxes[i].h/2.;
        float ymax = boxes[i].y + boxes[i].h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        float bx = xmin;
        float by = ymin;
        float bw = xmax - xmin;
        float bh = ymax - ymin;

        for(j = 0; j < classes; ++j){
            if (probs[i][j]) fprintf(fp, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n", image_id, coco_ids[j], bx, by, bw, bh, probs[i][j]);
        }
    }
}

void print_detector_detections(FILE **fps, char *id, box *boxes, float **probs, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = boxes[i].x - boxes[i].w/2.;
        float xmax = boxes[i].x + boxes[i].w/2.;
        float ymin = boxes[i].y - boxes[i].h/2.;
        float ymax = boxes[i].y + boxes[i].h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (probs[i][j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, probs[i][j],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void print_imagenet_detections(FILE *fp, int id, box *boxes, float **probs, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = boxes[i].x - boxes[i].w/2.;
        float xmax = boxes[i].x + boxes[i].w/2.;
        float ymin = boxes[i].y - boxes[i].h/2.;
        float ymax = boxes[i].y + boxes[i].h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            int class = j;
            if (probs[i][class]) fprintf(fp, "%d %d %f %f %f %f %f\n", id, j+1, probs[i][class],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void validate_detector(char *datacfg, char *cfgfile, char *weightfile)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "comp4_det_test_";
    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;

    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    if(0==strcmp(type, "coco")){
        snprintf(buff, 1024, "%s/coco_results.json", prefix);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    } else if(0==strcmp(type, "imagenet")){
        snprintf(buff, 1024, "%s/imagenet-detection.txt", prefix);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    } else {
        fps = calloc(classes, sizeof(FILE *));
        for(j = 0; j < classes; ++j){
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, base, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }


    box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
    float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .005;
    float nms = .45;

    int nthreads = 4;
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
            network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            get_region_boxes(l, w, h, thresh, probs, boxes, 0, map);
            if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, classes, nms);
            if (coco){
                print_cocos(fp, path, boxes, probs, l.w*l.h*l.n, classes, w, h);
            } else if (imagenet){
                print_imagenet_detections(fp, i+t-nthreads+1, boxes, probs, l.w*l.h*l.n, classes, w, h);
            } else {
                print_detector_detections(fps, id, boxes, probs, l.w*l.h*l.n, classes, w, h);
            }
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    for(j = 0; j < classes; ++j){
        if(fps) fclose(fps[j]);
    }
    if(coco){
        fseek(fp, -2, SEEK_CUR); 
        fprintf(fp, "\n]\n");
        fclose(fp);
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
}

void validate_detector_recall(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    list *plist = get_paths("data/voc.2007.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;

    int j, k;
    box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
    float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;

    float thresh = .001;
    float iou_thresh = .5;
    float nms = .4;

    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;

    for(i = 0; i < m; ++i){
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net.w, net.h);
        char *id = basecfg(path);
        network_predict(net, sized.data);
        get_region_boxes(l, 1, 1, thresh, probs, boxes, 1, 0);
        if (nms) do_nms(boxes, probs, l.w*l.h*l.n, 1, nms);

        char labelpath[4096];
        find_replace(path, "images", "labels", labelpath);
        find_replace(labelpath, "JPEGImages", "labels", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".JPEG", ".txt", labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        for(k = 0; k < l.w*l.h*l.n; ++k){
            if(probs[k][0] > thresh){
                ++proposals;
            }
        }
        for (j = 0; j < num_labels; ++j) {
            ++total;
            box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
            float best_iou = 0;
            for(k = 0; k < l.w*l.h*l.n; ++k){
                float iou = box_iou(boxes[k], t);
                if(probs[k][0] > thresh && iou > best_iou){
                    best_iou = iou;
                }
            }
            avg_iou += best_iou;
            if(best_iou > iou_thresh){
                ++correct;
            }
        }

        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
        free(id);
        free_image(orig);
        free_image(sized);
    }
}

void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/poster.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.4;
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
        layer l = net.layers[net.n-1];

        box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
        float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
        for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));

        float *X = sized.data;
        time=clock();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0);
        if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);
        save_image(im, "predictions");
        show_image(im, "predictions");

        free_image(im);
        free_image(sized);
        free(boxes);
        free_ptrs((void **)probs, l.w*l.h*l.n);
#ifdef OPENCV
        cvWaitKey(0);
        cvDestroyAllWindows();
#endif
        if (filename) break;
    }
}

void run_detector(int argc, char **argv)
{
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .24);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int clear = find_arg(argc, argv, "-clear");

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    if(0==strcmp(argv[2], "test")) test_detector(datacfg, cfg, weights, filename, thresh);
//     else if(0==strcmp(argv[2], "train")) train_detector(datacfg, cfg, weights, gpus, ngpus, clear);
		else if(0==strcmp(argv[2], "train")) train_detector(gpus, ngpus, clear);
    else if(0==strcmp(argv[2], "valid")) validate_detector(datacfg, cfg, weights);
    else if(0==strcmp(argv[2], "multivalid")) multivalidate_detector();
    else if(0==strcmp(argv[2], "recall")) validate_detector_recall(cfg, weights);
    else if(0==strcmp(argv[2], "demo")) {
        list *options = read_data_cfg(datacfg);
        int classes = option_find_int(options, "classes", 20);
        char *name_list = option_find_str(options, "names", "data/names.list");
        char **names = get_labels(name_list);
        demo(cfg, weights, thresh, cam_index, filename, names, classes, frame_skip, prefix);
    }
}

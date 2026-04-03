#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include<unistd.h>
#include<time.h>

#define NUM_USERS 10
#define NUM_SERVICES 2

typedef struct {
    int subscriber_count[NUM_SERVICES];
    pthread_mutex_t locks[NUM_SERVICES];
    char* service_names[NUM_SERVICES];
} StreamingPlatform;

StreamingPlatform platform;

void *subscribe_user(void *arg){
    int user_id = *(int *)arg;
    free(arg);
    int service_id = rand() % NUM_SERVICES;
    printf("[User %d] Attempting to subscribe to %s....\n", user_id, platform.service_names[service_id]);
    pthread_mutex_lock(&platform.locks[service_id]);
    usleep(10000);
    platform.subscriber_count[service_id]++;
    printf("[User %d] Sucess! Subscribed to %s. (Total subs %d)\n", user_id, platform.service_names[service_id], platform.subscriber_count[service_id]);
    pthread_mutex_unlock(&platform.locks[service_id]);
    return NULL;
}

int main(){
    pthread_t users[NUM_USERS];
    srand(time(NULL));

    platform.service_names[0] = "Apple Music";
    platform.service_names[1] = "Spotify Music";
    for(int i = 0; i < NUM_SERVICES; ++i){
        if(pthread_mutex_init(&platform.locks[i], NULL) != 0){
            perror("Mutex initialization failed");
            return 1;
        }
    }
    printf("--- Platform Open for Subscriptions ---\n\n");
    for(int i = 0; i < NUM_USERS; i++){
        int* user_id =  malloc(sizeof(int));
        *user_id = i + 1;
        if (pthread_create(&users[i], NULL, subscribe_user, user_id) != 0){
            perror("Thread Creation Failed!");
            return 1;
        }
    }
    for(int i = 0; i < NUM_USERS; i++){
        pthread_join(users[i], NULL);
    }
    printf("\n--- Final Subscription Report ---\n");
    for (int i = 0; i < NUM_SERVICES; i++){
        printf("%s: %d active subscriptions\n", platform.service_names[i], platform.subscriber_count[i]);
        pthread_mutex_destroy(&platform.locks[i]);
    }

    return 0;
}
#include<stdio.h>
#include<stdlib.h>
#include <string.h>
#include<unistd.h>
#include<pthread.h>

#define NUM_SUBSCRIBERS 3
#define NUM_MESSAGES 3

typedef struct {
    char message[256];
    int message_id;
    pthread_mutex_t lock;
    pthread_cond_t new_msg_cond;
} Topics;

Topics news_topics;

void *subscriber_routine(void *args){
    int id = *(int*)args;
    free(args);
    int last_read_id = 0;
    for(int i = 0; i < NUM_MESSAGES; i++){
        pthread_mutex_lock(&news_topics.lock);
        while(news_topics.message_id <= last_read_id){
            printf("[Subscriber %d] Waiting for news...\n", id);
            
            if (pthread_cond_wait(&news_topics.new_msg_cond, &news_topics.lock) != 0) {
                perror("[Subscriber] Error while conditional wait\n");
                return NULL;
            }
        }
        printf("[Subscriber %d] Received '%s' (MSG ID: %d)\n", id, news_topics.message, news_topics.message_id);
        last_read_id = news_topics.message_id;
        pthread_mutex_unlock(&news_topics.lock);
    }
    return NULL;
}

void *publisher_routine(void *args){
    for(int i = 1; i <= NUM_MESSAGES; i++){
        sleep(2);
        pthread_mutex_lock(&news_topics.lock);
        snprintf(news_topics.message, sizeof(news_topics.message), "Breaking News Broadcast #%d", i);
        news_topics.message_id = i;
        printf("\n-->[Publisher] Publishing: '%s'\n", news_topics.message);
        pthread_cond_broadcast(&news_topics.new_msg_cond);
        pthread_mutex_unlock(&news_topics.lock);
    }
    return NULL;
}

int main(){
    pthread_t publisher;
    pthread_t subscribers[NUM_SUBSCRIBERS];
    
    pthread_mutex_init(&news_topics.lock, NULL);
    pthread_cond_init(&news_topics.new_msg_cond, NULL);

    news_topics.message_id = 0;
    for(int i = 0; i < NUM_SUBSCRIBERS; i++){
        int *sub_id = malloc(sizeof(int));
        *sub_id = i + 1;
        if(pthread_create(&subscribers[i], NULL, subscriber_routine, sub_id) != 0){
            perror("[Error] While creating subscribers");
            return 1;
        }
    }
    usleep(500000);
    pthread_create(&publisher, NULL, publisher_routine, NULL);
    
    pthread_join(publisher, NULL);
    for(int i = 0; i < NUM_SUBSCRIBERS; i++){
        pthread_join(subscribers[i], NULL);
    }
    pthread_mutex_destroy(&news_topics.lock);
    pthread_cond_destroy(&news_topics.new_msg_cond);
    
    printf("Pub/Sub Broadcast Complete\n");
    return 0;
}
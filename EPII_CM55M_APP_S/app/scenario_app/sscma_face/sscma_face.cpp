/*
 * sscma_face.cpp
 *
 * SSCMA with Face Recognition - Main Entry Point
 *
 * This app extends SSCMA with face embedding capabilities.
 * Use AT+FACE=1 to enable face mode, AT+FACE=0 to return to normal mode.
 */

#include "sscma_face.h"
#include "sscma/main_task.hpp"

#include <cstdio>

static void app(void* arg) {
    sscma::main_task::run();
}

extern "C" int app_main(void) {
    puts("Build date: " __DATE__ " " __TIME__);
    puts("SSCMA Face Recognition Edition");

    if (xTaskCreate(app, "app", 20480, NULL, 3, NULL) != pdPASS) {
        puts("APP creation failed!");
        while (1) {
        }
    }

    vTaskStartScheduler();

    return 0;
}

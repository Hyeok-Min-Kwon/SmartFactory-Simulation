#include <WiFi.h>
#include <HTTPClient.h>
#include "esp_camera.h"
#include "constants.h"

// WiFi 설정
//const char* ssid = "";
//const char* password = "";

// 서버 설정
const char* serverUrl = "http://13.125.121.143:8000";
const char* captureCheckEndpoint = "/capture-request";  // 사진 요청 확인 엔드포인트
const char* uploadEndpoint = "/upload";                  // 사진 업로드 엔드포인트

// 폴링 간격 (밀리초)
unsigned long lastPollTime = 0;
const unsigned long pollInterval = 2000;  // 2초마다 서버 확인

// 카메라 핀 설정
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

void initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  
  // 이미지 크기 및 품질 설정
  config.frame_size = FRAMESIZE_VGA;  // 640x480
  config.jpeg_quality = 12;           
  config.fb_count = 2;
  config.grab_mode = CAMERA_GRAB_LATEST;
  
  // 카메라 초기화
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return;
  }
  
  Serial.println("✅ Camera initialized successfully");
}

// 서버에 사진 요청이 있는지 확인
bool checkCaptureRequest() {
  HTTPClient http;
  String url = String(serverUrl) + String(captureCheckEndpoint);

  http.begin(url);
  int httpCode = http.GET();

  if (httpCode == 200) {
    String response = http.getString();
    http.end();
    // 서버에서 "true" 또는 "1"을 반환하면 사진 촬영
    if (response == "true" || response == "1") {
      return true;
    }
  } else {
    Serial.printf("Check request failed, code: %d\n", httpCode);
  }

  http.end();
  return false;
}

// 사진 촬영 후 서버로 업로드
bool captureAndUpload() {
  Serial.println("\n📸 Capture request from server");

  unsigned long start = millis();

  // 이미지 촬영
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("❌ Camera capture failed");
    return false;
  }

  unsigned long captureTime = millis() - start;
  Serial.printf("✅ Photo captured!\n");
  Serial.printf("   Size: %d bytes\n", fb->len);
  Serial.printf("   Capture time: %lu ms\n", captureTime);

  // 서버로 이미지 업로드
  HTTPClient http;
  String url = String(serverUrl) + String(uploadEndpoint);

  http.begin(url);
  http.addHeader("Content-Type", "image/jpeg");

  int httpCode = http.POST(fb->buf, fb->len);

  esp_camera_fb_return(fb);

  if (httpCode == 200) {
    Serial.println("✅ Image uploaded successfully");
    http.end();
    return true;
  } else {
    Serial.printf("❌ Upload failed, code: %d\n", httpCode);
    http.end();
    return false;
  }
}

void setup() {
  Serial.begin(115200);
  Serial.println("\n\n🚀 ESP32-CAM Starting...");
  Serial.println("================================");

  // WiFi 연결
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  WiFi.setSleep(false);

  Serial.print("Connecting to WiFi");
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✅ WiFi Connected!");
    Serial.print("📍 IP Address: ");
    Serial.println(WiFi.localIP());
    Serial.printf("📶 Signal Strength: %d dBm\n", WiFi.RSSI());
  } else {
    Serial.println("\n❌ WiFi Connection Failed!");
    Serial.println("Check SSID and Password");
    return;
  }

  // 카메라 초기화
  initCamera();

  Serial.println("================================");
  Serial.printf("🌐 Server: %s\n", serverUrl);
  Serial.println("Waiting for capture requests...");
  Serial.println("================================\n");
}

void loop() {
  unsigned long currentTime = millis();

  // 폴링 간격마다 서버 확인
  if (currentTime - lastPollTime >= pollInterval) {
    lastPollTime = currentTime;

    // WiFi 연결 확인
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("WiFi disconnected, reconnecting...");
      WiFi.reconnect();
      return;
    }

    // 서버에 사진 요청이 있는지 확인
    if (checkCaptureRequest()) {
      captureAndUpload();
    }
  }
}
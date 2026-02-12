#include <WiFi.h>
#include <WebServer.h>
#include "esp_camera.h"

// WiFi 설정
//const char* ssid = "";       
//const char* password = "";

WebServer server(80);

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

void handleRoot() {
  String html = "<html><body style='font-family: Arial;'>";
  html += "<h1>ESP32-CAM Test</h1>";
  html += "<p><a href='/capture'><button style='padding:15px 30px; font-size:18px;'>📸 Capture Photo</button></a></p>";
  html += "<p><img src='/stream' width='640' height='480'></p>";
  html += "</body></html>";
  
  server.send(200, "text/html", html);
}

void handleCapture() {
  Serial.println("\n📸 Capture request received");
  
  unsigned long start = millis();
  
  // 이미지 촬영
  camera_fb_t * fb = esp_camera_fb_get();
  if(!fb) {
    Serial.println("❌ Camera capture failed");
    server.send(500, "text/plain", "Camera Error");
    return;
  }
  
  unsigned long elapsed = millis() - start;
  
  Serial.printf("✅ Photo captured!\n");
  Serial.printf("   Size: %d bytes\n", fb->len);
  Serial.printf("   Time: %lu ms\n", elapsed);
  
  // JPEG 이미지 전송
  server.sendHeader("Content-Disposition", "inline; filename=capture.jpg");
  server.send_P(200, "image/jpeg", (const char *)fb->buf, fb->len);
  
  esp_camera_fb_return(fb);
  
  Serial.println("   Image sent to client\n");
}

void handleStream() {
  // 실시간 미리보기용
  camera_fb_t * fb = esp_camera_fb_get();
  if(fb) {
    server.send_P(200, "image/jpeg", (const char *)fb->buf, fb->len);
    esp_camera_fb_return(fb);
  } else {
    server.send(500, "text/plain", "Stream Error");
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
  
  if(WiFi.status() == WL_CONNECTED) {
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
  
  // 웹서버 설정
  server.on("/", handleRoot);
  server.on("/capture", handleCapture);
  server.on("/stream", handleStream);
  
  server.begin();
  
  Serial.println("================================");
  Serial.println("🌐 Web Server Started");
  Serial.printf("Access at: http://%s\n", WiFi.localIP().toString().c_str());
  Serial.println("================================\n");
}

void loop() {
  server.handleClient();
}
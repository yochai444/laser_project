#include <driver/i2s.h>

#define I2S_BCK 15
#define I2S_WS  4
#define I2S_SD  2
#define SEL_PIN 16

#define TRIGGER_PIN 18
#define GROUND_PIN  19

#define PULSE_PIN 22
#define PULSE_FREQ_HZ 3000

#define I2S_PORT I2S_NUM_0
#define SAMPLE_RATE 16000
#define RECORD_SECONDS 20
#define TOTAL_SAMPLES (SAMPLE_RATE * RECORD_SECONDS)

hw_timer_t *timer = NULL;
volatile bool pulseState = false;
volatile bool generating = false;

void IRAM_ATTR onTimer() {
  if (generating) {
    pulseState = !pulseState;
    digitalWrite(PULSE_PIN, pulseState);
  }
}

void setup() {
  Serial.begin(921600);

  pinMode(SEL_PIN, OUTPUT);
  digitalWrite(SEL_PIN, LOW);

  pinMode(TRIGGER_PIN, INPUT_PULLUP);
  pinMode(GROUND_PIN, OUTPUT);
  digitalWrite(GROUND_PIN, LOW);

  pinMode(PULSE_PIN, OUTPUT);
  digitalWrite(PULSE_PIN, LOW);

  timer = timerBegin(PULSE_FREQ_HZ * 2);
  timerAttachInterrupt(timer, &onTimer);
  timerAlarm(timer, 1, true, 0);

  const i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 1024,
    .use_apll = false
  };

  const i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_BCK,
    .ws_io_num = I2S_WS,
    .data_out_num = -1,
    .data_in_num = I2S_SD
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);
}

void waitForGo() {
  while (true) {
    if (Serial.available()) {
      String cmd = Serial.readStringUntil('\n');
      cmd.trim();

      if (cmd == "GO") {
        return;
      }
    }
  }
}

void record_and_stream() {
  Serial.println("TRIGGERED");

  waitForGo();

  delay(20);

  size_t bytes_read;
  int32_t raw_sample;

  generating = true;
  pulseState = false;
  digitalWrite(PULSE_PIN, LOW);

  for (int i = 0; i < TOTAL_SAMPLES; i++) {
    i2s_read(I2S_PORT, &raw_sample, sizeof(raw_sample), &bytes_read, portMAX_DELAY);

    int16_t sample16 = (int16_t)(raw_sample >> 14);

    Serial.write((uint8_t)(sample16 & 0xFF));
    Serial.write((uint8_t)((sample16 >> 8) & 0xFF));
  }

  generating = false;
  digitalWrite(PULSE_PIN, LOW);

  Serial.flush();
}

void loop() {
  if (digitalRead(TRIGGER_PIN) == LOW) {
    delay(50);

    if (digitalRead(TRIGGER_PIN) == LOW) {
      record_and_stream();
    }

    while (digitalRead(TRIGGER_PIN) == LOW);
  }
}
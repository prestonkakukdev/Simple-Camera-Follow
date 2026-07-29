/*
 * camerafollow -- DIY pan/tilt head firmware
 *
 * Two stepper motors on STEP/DIR drivers (A4988, DRV8825, TMC2209...), driven
 * as continuous velocity. Speaks the simple line protocol used by
 * camerafollow.backends.serial_head:
 *
 *   V <pan> <tilt>\n    velocities in [-1, 1]   ->  (no reply)
 *   S\n                 immediate stop          ->  "OK STOP"
 *   H\n                 return to home (zero)   ->  "OK HOME"
 *   P\n                 ping                    ->  "OK <fw version>"
 *   Z\n                 set current position as home  ->  "OK ZERO"
 *
 * WIRING (adjust the pins below to taste)
 *   PAN  driver   STEP -> D2    DIR -> D5
 *   TILT driver   STEP -> D3    DIR -> D6
 *   both drivers  EN   -> D8    (active LOW; released when idle)
 *   driver GND    -> Arduino GND        <- do not skip this
 *   motor supply  12V, common ground with the Arduino
 *
 * Requires the AccelStepper library (Library Manager -> "AccelStepper").
 *
 * TWO SAFETY BEHAVIOURS, both deliberately in firmware rather than on the host,
 * because firmware is the side that keeps working when the host does not:
 *
 *  1. WATCHDOG. No command for WATCHDOG_MS and the motors stop. If the host
 *     process dies or the USB cable is pulled mid-pan, the camera stops instead
 *     of slewing into its end stop for the rest of the service.
 *
 *  2. ACCELERATION LIMIT. A stepper commanded to change speed instantly skips
 *     steps, and a skipped step is a permanently lost zero. The host limits
 *     acceleration too, but this is the layer that must not be bypassable.
 */

#include <AccelStepper.h>

// ---------------------------------------------------------------- pins ----
const uint8_t PAN_STEP  = 2;
const uint8_t PAN_DIR   = 5;
const uint8_t TILT_STEP = 3;
const uint8_t TILT_DIR  = 6;
const uint8_t ENABLE_PIN = 8;   // active LOW

// ------------------------------------------------------------- tuning ----
// Steps per second at full commanded velocity (1.0). Start low and raise it
// until the motion is as fast as you need; too high and the motor stalls with
// a buzz instead of turning.
const float PAN_MAX_SPS  = 1600.0f;
const float TILT_MAX_SPS =  900.0f;

// Steps per second per second. This is the ease-in/ease-out of the move.
const float PAN_ACCEL_SPS2  = 3000.0f;
const float TILT_ACCEL_SPS2 = 1800.0f;

// Soft limits in steps from home. Prevents winding the cable off the camera.
// Set to 0 to disable an axis limit.
const long PAN_LIMIT  = 0;
const long TILT_LIMIT = 4000;

const unsigned long WATCHDOG_MS = 300;
const unsigned long IDLE_RELEASE_MS = 5000;  // de-energise coils when parked

const char FW_VERSION[] = "camerafollow-pantilt 1.0";

// --------------------------------------------------------------- state ----
AccelStepper panMotor(AccelStepper::DRIVER, PAN_STEP, PAN_DIR);
AccelStepper tiltMotor(AccelStepper::DRIVER, TILT_STEP, TILT_DIR);

float panTarget = 0.0f, tiltTarget = 0.0f;   // commanded, normalised
float panSpeed  = 0.0f, tiltSpeed  = 0.0f;   // actual, steps/sec
unsigned long lastCommandMs = 0;
unsigned long lastUpdateUs = 0;
unsigned long stoppedSinceMs = 0;
bool homing = false;
bool enabled = false;

char lineBuf[48];
uint8_t lineLen = 0;

// ---------------------------------------------------------------- setup ---
void setup() {
  Serial.begin(115200);
  pinMode(ENABLE_PIN, OUTPUT);
  setEnabled(false);

  panMotor.setMaxSpeed(PAN_MAX_SPS);
  tiltMotor.setMaxSpeed(TILT_MAX_SPS);
  panMotor.setAcceleration(PAN_ACCEL_SPS2);
  tiltMotor.setAcceleration(TILT_ACCEL_SPS2);
  panMotor.setCurrentPosition(0);
  tiltMotor.setCurrentPosition(0);

  lastUpdateUs = micros();
  lastCommandMs = millis();
}

// ----------------------------------------------------------------- loop ---
void loop() {
  readSerial();

  unsigned long nowMs = millis();
  unsigned long nowUs = micros();
  float dt = (nowUs - lastUpdateUs) / 1000000.0f;
  if (dt <= 0.0f) dt = 0.000001f;
  lastUpdateUs = nowUs;

  // 1. watchdog
  if (nowMs - lastCommandMs > WATCHDOG_MS) {
    panTarget = 0.0f;
    tiltTarget = 0.0f;
  }

  if (homing) {
    runHoming();
    return;
  }

  // 2. acceleration-limited approach to the commanded speed
  panSpeed  = approach(panSpeed,  clampf(panTarget,  -1.0f, 1.0f) * PAN_MAX_SPS,
                       PAN_ACCEL_SPS2 * dt);
  tiltSpeed = approach(tiltSpeed, clampf(tiltTarget, -1.0f, 1.0f) * TILT_MAX_SPS,
                       TILT_ACCEL_SPS2 * dt);

  // 3. soft limits
  panSpeed  = applyLimit(panSpeed,  panMotor.currentPosition(),  PAN_LIMIT);
  tiltSpeed = applyLimit(tiltSpeed, tiltMotor.currentPosition(), TILT_LIMIT);

  bool moving = (fabs(panSpeed) > 1.0f) || (fabs(tiltSpeed) > 1.0f);
  if (moving) {
    setEnabled(true);
    stoppedSinceMs = 0;
  } else {
    if (stoppedSinceMs == 0) stoppedSinceMs = nowMs;
    if (nowMs - stoppedSinceMs > IDLE_RELEASE_MS) setEnabled(false);
  }

  panMotor.setSpeed(panSpeed);
  tiltMotor.setSpeed(tiltSpeed);
  panMotor.runSpeed();
  tiltMotor.runSpeed();
}

// -------------------------------------------------------------- helpers ---
void readSerial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (lineLen > 0) {
        lineBuf[lineLen] = '\0';
        handleLine(lineBuf);
        lineLen = 0;
      }
    } else if (lineLen < sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = c;
    } else {
      lineLen = 0;  // overlong garbage; resynchronise on the next newline
    }
  }
}

void handleLine(const char *line) {
  lastCommandMs = millis();
  switch (line[0]) {
    case 'V': case 'v': {
      float p = 0.0f, t = 0.0f;
      if (sscanf(line + 1, "%f %f", &p, &t) == 2) {
        panTarget = p;
        tiltTarget = t;
        homing = false;
      }
      break;                       // hot path: no reply, keeps the link quiet
    }
    case 'S': case 's':
      panTarget = tiltTarget = 0.0f;
      panSpeed = tiltSpeed = 0.0f;
      homing = false;
      Serial.println("OK STOP");
      break;
    case 'H': case 'h':
      homing = true;
      panTarget = tiltTarget = 0.0f;
      panMotor.moveTo(0);
      tiltMotor.moveTo(0);
      setEnabled(true);
      Serial.println("OK HOME");
      break;
    case 'Z': case 'z':
      panMotor.setCurrentPosition(0);
      tiltMotor.setCurrentPosition(0);
      Serial.println("OK ZERO");
      break;
    case 'P': case 'p':
      Serial.print("OK ");
      Serial.println(FW_VERSION);
      break;
    default:
      Serial.println("ERR");
      break;
  }
}

void runHoming() {
  panMotor.run();
  tiltMotor.run();
  if (panMotor.distanceToGo() == 0 && tiltMotor.distanceToGo() == 0) {
    homing = false;
    panSpeed = tiltSpeed = 0.0f;
  }
}

float approach(float current, float target, float maxDelta) {
  float delta = target - current;
  if (delta >  maxDelta) delta =  maxDelta;
  if (delta < -maxDelta) delta = -maxDelta;
  return current + delta;
}

float applyLimit(float speed, long position, long limit) {
  if (limit <= 0) return speed;                       // axis unlimited
  if (position >=  limit && speed > 0) return 0.0f;
  if (position <= -limit && speed < 0) return 0.0f;
  return speed;
}

float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

void setEnabled(bool on) {
  if (on == enabled) return;
  enabled = on;
  digitalWrite(ENABLE_PIN, on ? LOW : HIGH);   // drivers are active LOW
}

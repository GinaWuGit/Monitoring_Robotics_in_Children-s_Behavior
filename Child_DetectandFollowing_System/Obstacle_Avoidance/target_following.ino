//定義中心點座標
const int center_x = 320;
const int center_y = 240;

//設定速度誤差
const int correct = 9;
int final_speed;

//設定容忍值
const int tolerance_x = 50;
const int tolerance_y = 80;

//定義馬達輸出腳
const int motorA_1 = 3;
const int motorA_2 = 2;
const int motorB_1 = 5;
const int motorB_2 = 4;

// 設定速度控制輸出腳
const int enableA = 9;
const int enableB = 10;

void setup() {
  //設定輸出模式
  pinMode(motorA_1, OUTPUT);
  pinMode(motorA_2, OUTPUT);
  pinMode(motorB_1, OUTPUT);
  pinMode(motorB_2, OUTPUT);
  pinMode(enableA, OUTPUT);
  pinMode(enableB, OUTPUT);
  
  //初始化串口通訊
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n'); // 读取从Python发送的数据直到遇到换行符
    data.trim(); // 移除字符串首尾的空白字符

    // 解析接收到的坐标数据
    int delimiterIndex = data.indexOf(','); // 假設座標數據格視為 "x,y"
    if (delimiterIndex >= 0) {
      String xString = data.substring(0, delimiterIndex);
      String yString = data.substring(delimiterIndex + 1);
      int x = xString.toInt();
      int y = yString.toInt();

      // 計算在x軸與中心點的距離
      int distance_x = sqrt(pow(x - center_x, 2)); 
      // 計算在y軸與中心點的距離
      int distance_y = sqrt(pow(y - center_y, 2)); 

      int speed_x = map(distance_x, 0, 320, 200, 255);
      int speed_y = map(distance_y, 0, 240, 200, 255);
      //int speed_x = 245;
      //int speed_y = 245;
      // 根據與中心點的X軸的距，控制右轉、左轉或不轉
      if (distance_x > tolerance_x) {
        // 根据距离调整电机运动方向和速度
        if (x < center_x) {
          right(speed_x);
          Serial.print("turn left");
        } else {
          left(speed_x);
          Serial.print("turn right");
        }
      } else  if (distance_y > tolerance_y) {
        // 根據與中心點的Y軸的距，控制前進、倒退或不動
                if (y < center_y) {
                  backward(speed_y);
                  Serial.print("move backward");
                } else {
                  forward(speed_y);
                  Serial.print("move forward");
                }
              }


      // 向Python发送响应数据
      Serial.print("Received coordinates: ");
      Serial.print(x);
      Serial.print(", ");
      Serial.println(y);
    }
  }
}

//向前行駛
void forward(int speed) {
  digitalWrite(motorA_1, HIGH);
  digitalWrite(motorA_2, LOW);
  digitalWrite(motorB_1, HIGH);
  digitalWrite(motorB_2, LOW);
  
  // 設定速度
  final_speed = speed + correct;
  if (final_speed > 255)
    final_speed = 255;
  analogWrite(enableA, speed);
  analogWrite(enableB, final_speed);
}

//倒退
void backward(int speed) {
  digitalWrite(motorA_1, LOW);
  digitalWrite(motorA_2, HIGH);
  digitalWrite(motorB_1, LOW);
  digitalWrite(motorB_2, HIGH);
  
  // 設定速度
  final_speed = speed + correct;
  if (final_speed > 255)
    final_speed = 255;
  analogWrite(enableA, speed);
  analogWrite(enableB, final_speed);
}

// 左轉
void left(int speed) {
  digitalWrite(motorA_1, LOW);
  digitalWrite(motorA_2, HIGH);
  digitalWrite(motorB_1, HIGH);
  digitalWrite(motorB_2, LOW);
  
  // 設定速度
  final_speed = speed + correct;
  if (final_speed > 255)
    final_speed = 255;
  analogWrite(enableA, speed);
  analogWrite(enableB, final_speed);
}

// 右轉
void right(int speed) {
  digitalWrite(motorA_1, HIGH);
  digitalWrite(motorA_2, LOW);
  digitalWrite(motorB_1, LOW);
  digitalWrite(motorB_2, HIGH);
  
  // 設定速度
  final_speed = speed + correct;
  if (final_speed > 255)
    final_speed = 255;
  analogWrite(enableA, speed);
  analogWrite(enableB, final_speed);
}

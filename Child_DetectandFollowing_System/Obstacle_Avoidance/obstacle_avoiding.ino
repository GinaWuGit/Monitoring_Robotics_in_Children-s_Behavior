 // 前方30cm有障礙物,車子停止,偵測兩方路況,決定左轉彎或是右轉彎
 // 改進:轉彎弧度與馬力輸出設定調整到兩輪能直線前進
 
#include <Servo.h>
#include <RPLidar.h>

RPLidar lidar;  // 创建一个名为lidar的雷达驱动实例 

#define In1 7   //右馬達
#define In2 8
#define ENA 6
#define In3 12  //左馬達
#define In4 13
#define ENB 11
#define RPLIDAR_MOTOR 3   // The PWM pin for control the speed of RPLIDAR's motor.
                          //This pin should connected with the RPLIDAR's MOTOCTRL
                          
unsigned long d;    // 前方障礙物距離
const int leftSpeed = 115;   //左輪轉速<經過實際測試,若左右輪轉速相同時,自走車無法走直線>
const int rightSpeed =115;   //右輪轉速


void setup() {
  // bind the RPLIDAR driver to the arduino hardware serial
  lidar.begin(Serial1);
  Serial.begin(115200);
  
  //馬達與光達腳位設定
  pinMode(In1,OUTPUT); //右馬達
  pinMode(In2,OUTPUT);
  pinMode(In3,OUTPUT); //左馬達
  pinMode(In4,OUTPUT);
  pinMode(ENA,OUTPUT);
  pinMode(ENB,OUTPUT);
  pinMode(RPLIDAR_MOTOR, OUTPUT);
}

// 後退
void backward(){
   // 右馬達 後退
    digitalWrite(In1,HIGH);
    digitalWrite(In2,LOW);
    analogWrite(ENA,rightSpeed);
    // 左馬達 後退
    digitalWrite(In3,HIGH);
    digitalWrite(In4,LOW);
    analogWrite(ENB,leftSpeed);
  }

// 前進
 void forward(){
    // 右馬達 前進
    digitalWrite(In1,LOW);
    digitalWrite(In2,HIGH);
    analogWrite(ENA,rightSpeed);
    // 左馬達 前進
    digitalWrite(In3,LOW);
    digitalWrite(In4,HIGH);
    analogWrite(ENB,leftSpeed);
  }

 // 左轉
 void turnLeft(){
   // 右馬達 前進
    digitalWrite(In1,LOW);
    digitalWrite(In2,HIGH);
    analogWrite(ENA,rightSpeed);
    // 左馬達 後退
    digitalWrite(In3,HIGH);
    digitalWrite(In4,LOW);
    analogWrite(ENB,leftSpeed);
  }

  // 右轉
 void turnRight(){
   // 右馬達 後退
    digitalWrite(In1,HIGH);
    digitalWrite(In2,LOW);
    analogWrite(ENA,rightSpeed);
    // 左馬達 前進
    digitalWrite(In3,LOW);
    digitalWrite(In4,HIGH);
    analogWrite(ENB,leftSpeed);
  }

  //停止 
  void motoStop(){
     // 右馬達停止
    digitalWrite(In1,LOW);
    digitalWrite(In2,LOW);
    analogWrite(ENA,rightSpeed);
    // 左馬達停止
    digitalWrite(In3,LOW);
    digitalWrite(In4,LOW);
    analogWrite(ENB,leftSpeed);
   }

---------------------------------------------------------------------------------------------
void loop() {
  if (IS_OK(lidar.waitPoint())) {
    float distance = lidar.getCurrentPoint().distance;  //distance value in mm unit
                                                        
    float angle    = lidar.getCurrentPoint().angle;     //anglue value in degree
                                                        
    bool  startBit = lidar.getCurrentPoint().startBit;  //whether this point is belong to a new scan
                                                        
    byte  quality  = lidar.getCurrentPoint().quality;   //quality of the current measurement
                                                        
    //- 1 -
    // perform data processing here...
    // Output all data in the serial port  

    //    for(int i = 0;i < 6 - String(angle).length(); i++){
    //      Serial.print(" ");
    //    }
    //    Serial.print(String(angle));
    //    Serial.print(" | ");
    //    for(int i = 0;i < 8 - String(distance).length(); i++){
    //      Serial.print(" ");
    //    }
    //    Serial.print(distance);
    //    Serial.print(" | ");
    //    Serial.print(startBit);
    //    Serial.print(" | ");
    //    for(int i = 0;i < 2 - String(quality).length(); i++){
    //      Serial.print(" ");
    //    }
    //    Serial.println(quality);

    // - 2 - 
    // Output the specified angle data
    // 输出指定角度
    if(angle > 0 and angle < 360 ){
      Serial.print(angle);
      Serial.print(" | ");
      Serial.println(distance);
    }
  } else {
    analogWrite(RPLIDAR_MOTOR, 0); //stop the rplidar motor
                                   //停止rplidar马达

    // try to detect RPLIDAR...
    // 尝试检测RPLIDAR... 
    rplidar_response_device_info_t info;
    if (IS_OK(lidar.getDeviceInfo(info, 100))) {
       // detected...
       // 检测到
       lidar.startScan();

       // start motor rotating at max allowed speed
       // 启动电机以最大允许速度旋转
       analogWrite(RPLIDAR_MOTOR, 255);
       delay(1000);
    }
  }
  
 //  如果前方30cm處有障礙物,進入判斷模式決定行進方式
 if(distance<=30) {
    motoStop() ; //自走車停止前進
    delay(500);
    if(0<angle<90){
      turnLeft();
      delay(500);
    }else if(90<angle<180){
      turnRight();
      delay(500);
    }
    // 如果左邊空間大且障礙物距離超過30cm以上 ---> 左轉彎後繼續前進
    if( (left_d>right_d) && (left_d>30)) { //左邊有空間
        turnLeft() ;
        delay(350) ;
        forward() ;
     } else if( (right_d>=left_d) && (right_d>30)) { // 右邊空間大且右邊障礙物距離大於30cm以上 -->右轉彎後前進
        turnRight() ;
         delay(350) ;
         forward() ;
      } else {  // 前,左,右障礙物距離都小於30公分 --->後退->轉彎->前進
         backward() ;
         delay(1500) ;
         turnRight() ;
         delay(350) ;
         forward() ; 
       }   
  }
   delay(30) ;
}

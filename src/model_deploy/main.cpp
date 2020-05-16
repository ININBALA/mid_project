#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"
#include "mbed.h"
#include <cmath>
#include "DA7212.h"
#include "uLCD_4DGL.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#define bufferLength (32)
#define signalLength (1024)

DA7212 audio;
int16_t waveform[kAudioTxBufferSize];
// Return the result of the last prediction
uLCD_4DGL uLCD(D1, D0, D2);
Serial pc(USBTX, USBRX);

InterruptIn button1(SW2);
InterruptIn button2(SW3);
EventQueue queue1(32 * EVENTS_EVENT_SIZE);
EventQueue queue2(32 * EVENTS_EVENT_SIZE);
Thread t1;
Thread t2;
int idC = 0;
int state = 0;
int flag = 0;
int song[42] = {
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261,
  392, 392, 349, 349, 330, 330, 294,
  392, 392, 349, 349, 330, 330, 294,
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261};
int noteLength[42] = {
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2};
int songnum = 0;
char serialInBuffer[bufferLength];
DigitalOut green_led(LED2);
int serialCount = 0;
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
  //EventQueue queue;
  // Whether we should clear the buffer next time we fetch data
bool should_clear_buffer = false;
bool got_data = false;
  // The gesture index of the prediction
int gesture_index;
  // Set up logging.
static tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
static tflite::MicroOpResolver<6> micro_op_resolver;
static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
tflite::MicroInterpreter* interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
TfLiteTensor* model_input = interpreter->input(0);

int input_length = model_input->bytes / sizeof(float);
TfLiteStatus setup_status = SetupAccelerometer(error_reporter);

int PredictGesture(float* output) {
  // How many times the most recent gesture has been matched in a row
  static int continuous_count = 0;
  // The result of the last prediction
  static int last_predict = -1;

  // Find whichever output has a probability > 0.8 (they sum to 1)
  int this_predict = -1;
  for (int i = 0; i < label_num; i++) {
    if (output[i] > 0.8) this_predict = i;
  }

  // No gesture was detected above the threshold
  if (this_predict == -1) {
    continuous_count = 0;
    last_predict = label_num;
    return label_num;
  }

  if (last_predict == this_predict) {
    continuous_count += 1;
  } else {
    continuous_count = 0;
  }
  last_predict = this_predict;

  // If we haven't yet had enough consecutive matches for this gesture,
  // report a negative result
  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
    return label_num;
  }
  // Otherwise, we've seen a positive result, so clear all our variables
  // and report it
  continuous_count = 0;
  last_predict = -1;

  return this_predict;
}
/*void stopPlayNoteC(void) {
  //queue1.cancel(idC);
  audio.spk.pause();
  //pc.printf("%d\n",idC);
}*/
void playNote(int freq)
{
  for(int i = 0; i < kAudioTxBufferSize; i++)
  {
    waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency / freq)) * ((1<<16) - 1));
  }
  audio.spk.play(waveform, kAudioTxBufferSize);
}

void play(){
  audio.spk.play();
  uLCD.printf("\nASS\n");
   for(int i = 0; i < 42; i++)
  {
    int length = noteLength[i];
    while(length--)
    {
      // the loop below will play the note for the duration of 1s
      for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)
      {
        queue1.call(playNote, song[i]);
      }
      if(length < 1) wait(1.0);
    }
  }  
}
void forward(void){
  uLCD.cls(); 
  uLCD.printf("\nforward\n");
}

void backward(void){
  uLCD.cls(); 
  uLCD.printf("\nbackward\n");
}
void select(void){
  uLCD.cls();
  uLCD.printf("\nselect\n");
}
void modeSelect(){
 int i = 0, s = 0;
 uLCD.cls();
 uLCD.printf("\nselect\n");
 button2.rise(queue2.event(play));
 while(i < 1){
   if(button1 == 0){
     i = 1;
     //playNoteC();
   }
    switch(state){
      case 0:
       forward();
        break;
      case 1:
        backward();
        break;
      case 3:
        select();
        break;
      default:
        break;
    }
  }
}

void loadSignal(void)
{
  green_led = 0;
  int i = 0, flag = 0;
  serialCount = 0;
  audio.spk.pause();
  while(i < 42)
  {
    if(pc.readable())
    {
      serialInBuffer[serialCount] = pc.getc();
      serialCount++;
      if(serialCount == 3)
      {//pc.printf("%f\n",signal[0]);
        serialInBuffer[serialCount] = '\0';
        song[i] = (int) atof(serialInBuffer);
        uLCD.printf("\n%d\n", song[i]);
        //flag = (int) atof(serialInBuffer);
        //pc.printf("%d\n\r",signal[i]);
        //pc.printf("%d\n",flag);
        serialCount = 0;
        i++;
      }
    }
  }
  green_led = 1;
}
void confirm(void){
  switch(state){
      case 0:
        uLCD.cls();
        if(songnum < 2)
          songnum++;
        pc.printf("%d", songnum);
        uLCD.printf("\nLoad song%d...\n", songnum);
        loadSignal();
        uLCD.printf("\nplay song%d\n", songnum);
        play();
        uLCD.cls();
        uLCD.printf("\nTake a rest...\n");
        state = 3;
        break;
      case 1:
        uLCD.cls();
        if(songnum > 0)
          songnum--;
        pc.printf("%d", songnum);
        uLCD.printf("\nLoad song%d...\n", songnum);
        loadSignal();
        uLCD.printf("\nplay song%d\n", songnum);
        play();
        uLCD.cls();
        uLCD.printf("\nTake a rest...\n");
        state = 3;
        break;
        break;
      case 2:
        audio.spk.pause();
        uLCD.cls();
        uLCD.printf("\nselect your song\n");
        while (true) {
          // Attempt to read new data from the accelerometer
          got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                      input_length, should_clear_buffer);
          // If there was no new data,
          // don't try to clear the buffer again and wait until next time
          if (!got_data) {
            should_clear_buffer = false;
            continue;
          }
          // Run inference, and report any error
          TfLiteStatus invoke_status = interpreter->Invoke();
          if (invoke_status != kTfLiteOk) {
            error_reporter->Report("Invoke failed on index: %d\n", begin_index);
            continue;
          }
          // Analyze the results to obtain a prediction
          gesture_index = PredictGesture(interpreter->output(0)->data.f);

          // Clear the buffer next time we read data
          should_clear_buffer = gesture_index < label_num;

          // Produce an output
          if (gesture_index < label_num) {
            //error_reporter->Report(config.output_message[gesture_index]);
            if(songnum < 2)
              songnum++;
            else
              songnum = 0;
            uLCD.cls();
            uLCD.printf("\nselect your song\n");
            uLCD.printf("\nsong%d\n", songnum);
          }
          if(button2 == 0)
            break;
        }
        uLCD.printf("\nLoad song%d...\n", songnum);
        pc.printf("%d", songnum);
        loadSignal();
        uLCD.printf("\nplay song%d\n", songnum);
        play();
        uLCD.cls();
        uLCD.printf("\nTake a rest...\n");
        state = 3;
        break;
      default:
        break;
      } 
} 
void DNN(){
  audio.spk.pause();
  uLCD.cls();
  uLCD.printf("\nmode\n");
  int selectsong = 0;
  button2.rise(queue2.event(confirm));
  while (true) {
    // Attempt to read new data from the accelerometer
    got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                 input_length, should_clear_buffer);
    // If there was no new data,
    // don't try to clear the buffer again and wait until next time
    if (!got_data) {
      should_clear_buffer = false;
      continue;
    }
    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed on index: %d\n", begin_index);
      continue;
    }
    // Analyze the results to obtain a prediction
    gesture_index = PredictGesture(interpreter->output(0)->data.f);

    // Clear the buffer next time we read data
    should_clear_buffer = gesture_index < label_num;

    // Produce an output
    if (gesture_index < label_num) {
      //error_reporter->Report(config.output_message[gesture_index]);
      if(state < 2)
        state++;
      else
        state = 0;
        switch(state){
        case 0:
        forward();
          break;
        case 1:
          backward();
          break;
        case 2:
          select();
          break;
        default:
          break;
        }/*
      else
      {
        if(songnum < 2)
          songnum++;
        else
          songnum = 0;
        uLCD.printf("\nsong%d\n", songnum);
      }*/
    }
    if(button2 == 0)
      break;/*
    else if(button2 == 0 && state == 2){
      selectsong = 1;
      wait(0.5);
    }
    else if(button2 == 0 && selectsong == 1)
      break;*/
  }
}
int main(int argc, char* argv[]) {
  // Create an area of memory to use for input, output, and intermediate arrays.
  // The size of this will depend on the model you're using, and may need to be
  // determined by experimentation.
 
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return -1;
  }
  
  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                               tflite::ops::micro::Register_MAX_POOL_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                               tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                               tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());
                               
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                               tflite::ops::micro::Register_RESHAPE(),1);
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                               tflite::ops::micro::Register_CONV_2D(),1);

  // Build an interpreter to run the model with
  
  // Obtain pointer to the model's input tensor
  interpreter->AllocateTensors();
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != config.seq_length) ||
      (model_input->dims->data[2] != kChannelNumber) ||
      (model_input->type != kTfLiteFloat32)) {
   // error_reporter->Report("Bad input tensor pabutton1.rise(queue1.event(backward));rameters in model");
    return -1;
  }


  if (setup_status != kTfLiteOk) {
   // error_reporter->Report("Set up failed\n");
    return -1;
  }

  //error_reporter->Report("Set up successful...\n");
  /*for(int i = 0; i < 42; i++)
  {
     // the loop below will play the note for the duration of 1s
    for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)
    {
      queue.call(playNote, song[i]);
    }
    wait(1.0);
  }*/
  uLCD.printf("\nPlay Song\n");
  t1.start(callback(&queue1, &EventQueue::dispatch_forever));
  button1.rise(queue1.event(DNN));
  t2.start(callback(&queue2, &EventQueue::dispatch_forever));
  
}
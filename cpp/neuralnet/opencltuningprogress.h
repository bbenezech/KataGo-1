#ifndef NEURALNET_OPENCL_TUNING_PROGRESS_H_
#define NEURALNET_OPENCL_TUNING_PROGRESS_H_

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

// Thread-safe shared struct for surfacing OpenCL tuning progress from the
// tuner thread to an external heartbeat thread.  The tuner writes via
// updateStage / updateConfigProgress / complete; the heartbeat reads via
// snapshot().

struct OpenCLTuningProgress {
  struct Snapshot {
    bool active;
    std::string currentStage;
    std::string stageStatus;      // "in_progress", "done", "skipped"
    int configsTested;
    int configsTotal;
    int64_t stageDurationMs;
    std::vector<std::string> stagesCompleted;
    std::vector<std::string> stagesRemaining;
    std::string deviceName;
    bool fp16Storage;
    bool fp16Compute;
    bool fp16TensorCores;
  };

  // ── Writer API (called from tuner thread) ──

  void setDeviceInfo(const std::string& name, bool fp16Stor, bool fp16Comp, bool fp16TC) {
    std::lock_guard<std::mutex> lock(mu_);
    deviceName_ = name;
    fp16Storage_ = fp16Stor;
    fp16Compute_ = fp16Comp;
    fp16TensorCores_ = fp16TC;
  }

  void setStageOrder(const std::vector<std::string>& allStages) {
    std::lock_guard<std::mutex> lock(mu_);
    allStages_ = allStages;
  }

  void beginStage(const std::string& stageName) {
    std::lock_guard<std::mutex> lock(mu_);
    active_ = true;
    currentStage_ = stageName;
    stageStatus_ = "in_progress";
    configsTested_ = 0;
    configsTotal_ = 0;
    stageStartedAt_ = std::chrono::steady_clock::now();
  }

  void updateConfigProgress(int tested, int total) {
    std::lock_guard<std::mutex> lock(mu_);
    configsTested_ = tested;
    configsTotal_ = total;
  }

  void completeStage(const std::string& stageName, const std::string& status) {
    std::lock_guard<std::mutex> lock(mu_);
    if(status == "done" || status == "skipped") {
      stagesCompleted_.push_back(stageName);
    }
    if(currentStage_ == stageName) {
      stageStatus_ = status;
    }
  }

  void complete() {
    std::lock_guard<std::mutex> lock(mu_);
    active_ = false;
  }

  // ── Reader API (called from heartbeat thread) ──

  Snapshot snapshot() const {
    std::lock_guard<std::mutex> lock(mu_);
    Snapshot snap;
    snap.active = active_;
    snap.currentStage = currentStage_;
    snap.stageStatus = stageStatus_;
    snap.configsTested = configsTested_;
    snap.configsTotal = configsTotal_;
    snap.stageDurationMs = active_ ? std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - stageStartedAt_
    ).count() : 0;
    snap.stagesCompleted = stagesCompleted_;
    snap.deviceName = deviceName_;
    snap.fp16Storage = fp16Storage_;
    snap.fp16Compute = fp16Compute_;
    snap.fp16TensorCores = fp16TensorCores_;

    // Build stagesRemaining from allStages_ minus stagesCompleted_
    for(const auto& stage : allStages_) {
      bool isCompleted = false;
      for(const auto& completed : stagesCompleted_) {
        if(completed == stage) {
          isCompleted = true;
          break;
        }
      }
      // Also exclude the current in-progress stage
      if(!isCompleted && stage != currentStage_) {
        snap.stagesRemaining.push_back(stage);
      }
    }
    return snap;
  }

  // ── Thread-local singleton pointer ──

  static OpenCLTuningProgress* getCurrent() { return current_; }
  static void setCurrent(OpenCLTuningProgress* p) { current_ = p; }

private:
  mutable std::mutex mu_;
  bool active_ = false;
  std::string currentStage_;
  std::string stageStatus_;
  int configsTested_ = 0;
  int configsTotal_ = 0;
  std::chrono::steady_clock::time_point stageStartedAt_;
  std::vector<std::string> stagesCompleted_;
  std::vector<std::string> allStages_;
  std::string deviceName_;
  bool fp16Storage_ = false;
  bool fp16Compute_ = false;
  bool fp16TensorCores_ = false;

  static thread_local OpenCLTuningProgress* current_;
};

#endif  // NEURALNET_OPENCL_TUNING_PROGRESS_H_

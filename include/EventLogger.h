#ifndef EVENTLOGGER_H
#define EVENTLOGGER_H

#include <fstream>
#include <mutex>
#include <string>

namespace ORB_SLAM3
{

class EventLogger
{
public:
    static EventLogger& Instance();

    void Open(const std::string& filename);
    void OpenKeyframes(const std::string& filename);
    void OpenFeatures(const std::string& filename);

    // Backwards-compatible simple event logger.
    void Log(long long frame_id,
             double dataset_time,
             const std::string& module,
             const std::string& event,
             const std::string& details);

    // Improved event logger.
    void LogEvent(long long frame_id,
                  double dataset_time,
                  const std::string& module,
                  const std::string& event,
                  int state,
                  const std::string& state_name,
                  long long current_kf,
                  long long matched_kf,
                  int matches_inliers,
                  const std::string& details);

    void LogKeyFrame(long long keyframe_id,
                     long long frame_id,
                     double dataset_time,
                     int matches_inliers,
                     int total_keypoints,
                     int tracked_mappoints);
    void LogFeature(long long frame_id,
                double dataset_time,
                int feature_idx,
                float x,
                float y,
                int octave,
                float response,
                int has_mappoint,
                int is_outlier,
                long long mappoint_id);

    void Close();

private:
    EventLogger() = default;

    std::string EscapeCsv(const std::string& text);

    std::ofstream mEventFile;
    std::ofstream mKeyframeFile;
    std::ofstream mFeatureFile;
    std::mutex mMutex;
};

}

#endif

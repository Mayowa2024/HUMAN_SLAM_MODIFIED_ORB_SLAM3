#include "EventLogger.h"

namespace ORB_SLAM3
{

EventLogger& EventLogger::Instance()
{
    static EventLogger instance;
    return instance;
}

void EventLogger::Open(const std::string& filename)
{
    std::lock_guard<std::mutex> lock(mMutex);

    if (mEventFile.is_open())
        mEventFile.close();

    mEventFile.open(filename);

    if (mEventFile.is_open())
    {
        mEventFile << "frame_id,dataset_time,module,event,state,state_name,current_kf,matched_kf,matches_inliers,details\n";
    }
}

void EventLogger::OpenKeyframes(const std::string& filename)
{
    std::lock_guard<std::mutex> lock(mMutex);

    if (mKeyframeFile.is_open())
        mKeyframeFile.close();

    mKeyframeFile.open(filename);

    if (mKeyframeFile.is_open())
    {
        mKeyframeFile << "keyframe_id,frame_id,dataset_time,matches_inliers,total_keypoints,tracked_mappoints\n";
    }
}

void EventLogger::OpenFeatures(const std::string& filename)
{
    std::lock_guard<std::mutex> lock(mMutex);

    if (mFeatureFile.is_open())
        mFeatureFile.close();

    mFeatureFile.open(filename);

    if (mFeatureFile.is_open())
    {
        mFeatureFile << "frame_id,dataset_time,feature_idx,x,y,octave,response,has_mappoint,is_outlier,mappoint_id\n";
    }
}

std::string EventLogger::EscapeCsv(const std::string& text)
{
    std::string escaped;

    for (char c : text)
    {
        if (c == '"')
            escaped += "\"\"";
        else
            escaped += c;
    }

    return escaped;
}

void EventLogger::Log(long long frame_id,
                      double dataset_time,
                      const std::string& module,
                      const std::string& event,
                      const std::string& details)
{
    LogEvent(
        frame_id,
        dataset_time,
        module,
        event,
        -1,
        "N/A",
        -1,
        -1,
        -1,
        details
    );
}

void EventLogger::LogEvent(long long frame_id,
                           double dataset_time,
                           const std::string& module,
                           const std::string& event,
                           int state,
                           const std::string& state_name,
                           long long current_kf,
                           long long matched_kf,
                           int matches_inliers,
                           const std::string& details)
{
    std::lock_guard<std::mutex> lock(mMutex);

    if (mEventFile.is_open())
    {
        mEventFile << frame_id << ","
                   << dataset_time << ","
                   << module << ","
                   << event << ","
                   << state << ","
                   << state_name << ","
                   << current_kf << ","
                   << matched_kf << ","
                   << matches_inliers << ","
                   << "\"" << EscapeCsv(details) << "\""
                   << "\n";
    }
}

void EventLogger::LogKeyFrame(long long keyframe_id,
                              long long frame_id,
                              double dataset_time,
                              int matches_inliers,
                              int total_keypoints,
                              int tracked_mappoints)
{
    std::lock_guard<std::mutex> lock(mMutex);

    if (mKeyframeFile.is_open())
    {
        mKeyframeFile << keyframe_id << ","
                      << frame_id << ","
                      << dataset_time << ","
                      << matches_inliers << ","
                      << total_keypoints << ","
                      << tracked_mappoints
                      << "\n";
    }
}

void EventLogger::LogFeature(long long frame_id,
                             double dataset_time,
                             int feature_idx,
                             float x,
                             float y,
                             int octave,
                             float response,
                             int has_mappoint,
                             int is_outlier,
                             long long mappoint_id)
{
    std::lock_guard<std::mutex> lock(mMutex);

    if (mFeatureFile.is_open())
    {
        mFeatureFile << frame_id << ","
                     << dataset_time << ","
                     << feature_idx << ","
                     << x << ","
                     << y << ","
                     << octave << ","
                     << response << ","
                     << has_mappoint << ","
                     << is_outlier << ","
                     << mappoint_id
                     << "\n";
    }
}
void EventLogger::Close()
{
    std::lock_guard<std::mutex> lock(mMutex);

    if (mEventFile.is_open())
        mEventFile.close();

    if (mKeyframeFile.is_open())
        mKeyframeFile.close();
    if (mFeatureFile.is_open())
    mFeatureFile.close();
}

}

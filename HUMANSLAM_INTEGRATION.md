# ORB-SLAM3 HumanSLAM integration

This repository is based on
[UZ-SLAMLab/ORB_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) and preserves
the upstream history, licence and attribution.

The HumanSLAM modifications add:

- a semantic tracking snapshot containing the current frame, reference
  keyframe, Atlas map, tracking state and inlier count;
- a thread-safe semantic candidate interface in `System` and `Tracking`;
- resolution of HumanSLAM map/keyframe identifiers against the live Atlas;
- ORB matching, PnP and pose-optimised semantic relocalisation candidates;
- conventional ORB bag-of-words fallback when semantic candidates fail;
- a separate semantic cross-map proposal queue in `LoopClosing`;
- semantic prioritisation of cross-map candidates without weakening ORB-SLAM3
  geometric thresholds;
- Sim(3), projection and three-keyframe temporal verification before standard
  ORB-SLAM3 Atlas fusion; and
- diagnostic event, keyframe and feature logging used during evaluation.

HumanSLAM proposes candidate places. ORB-SLAM3 remains responsible for pose
estimation, geometric verification, map transformation and optimisation.

The companion semantic inference repository is:

- <https://github.com/Mayowa2024/HumanSLAM>

The ROS 2 wrapper and custom message packages are maintained separately from
the ORB-SLAM3 core.

## Build

Follow the upstream ORB-SLAM3 dependency instructions, then build:

```bash
chmod +x build.sh
./build.sh
```

The TensorRT semantic models are not part of this repository. They belong to
the companion HumanSLAM project.

## Upstream acknowledgement

ORB-SLAM3 was developed by Carlos Campos, Richard Elvira, Juan J. Gómez
Rodríguez, José M. M. Montiel and Juan D. Tardós. Cite the upstream ORB-SLAM3
publication and comply with the repository's GPLv3 licence when using or
redistributing this modified version.

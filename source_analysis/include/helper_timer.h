#pragma once
// Analysis-only timer declarations. Timing is unknown, not measured.
struct StopWatchInterface {};
void sdkCreateTimer(StopWatchInterface**);
void sdkStartTimer(StopWatchInterface**);
void sdkStopTimer(StopWatchInterface**);
float sdkGetAverageTimerValue(StopWatchInterface**);

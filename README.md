# Blink-Fatigue-detection

# 👁️ BlinkSense

A real-time blink counter and fatigue detector using OpenCV and MediaPipe. It tracks both-eye blinks, detects extended eye closures (signs of fatigue), and avoids false counts due to head movements.

---

## 🔍 Features

- ✅ Real-time **both-eye blink detection**
- ⏱️ **Fatigue detection**: triggers alert if eyes closed for > 5 seconds
- 🧠 **Head movement filtering**: skips frame if user turns or shakes head
- ⚡ Smooth EAR (Eye Aspect Ratio) detection using landmark smoothing

---

## 🛠️ Technologies Used

- Python 3
- OpenCV
- MediaPipe FaceMesh
- Math, Time libraries

---

## 🚀 Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/your-username/BlinkSense.git
cd BlinkSense

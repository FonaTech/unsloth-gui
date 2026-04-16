"""
core/monitor.py
训练指标收集与状态管理。
训练线程通过 put() 推送事件，Gradio UI 通过 poll() 拉取并刷新。
"""

import queue
import time
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any


@dataclass
class TrainingMetrics:
    step: int
    loss: Optional[float] = None
    eval_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    epoch: Optional[float] = None
    speed: Optional[float] = None       # samples/s
    gpu_mem_gb: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


class TrainingMonitor:
    """线程安全的训练监控器，连接训练子线程与 Gradio 主线程。"""

    def __init__(self):
        self._queue: queue.Queue = queue.Queue(maxsize=10000)
        self._lock = threading.Lock()
        # 指标历史
        self.history: List[TrainingMetrics] = []
        self.current: Optional[TrainingMetrics] = None
        # 检查点
        self.checkpoints: List[str] = []
        # 状态
        self.status: str = "idle"   # idle|loading|running|paused|stopped|finished|error
        self.error_message: str = ""
        # 进度
        self.total_steps: int = 0
        self.completed_steps: int = 0
        self.start_time: Optional[float] = None
        # 日志行（最近 500 行）
        self.log_lines: List[str] = []
        self._max_log_lines = 500

    # ── 写入（训练线程调用） ──────────────────────────────────────────

    def put(self, event: Dict[str, Any]) -> None:
        """从训练线程推送事件到队列（非阻塞，满则丢弃）。"""
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            pass

    def reset(self) -> None:
        """重置监控器状态（开始新训练前调用）。"""
        with self._lock:
            self.history.clear()
            self.current = None
            self.checkpoints.clear()
            self.status = "idle"
            self.error_message = ""
            self.total_steps = 0
            self.completed_steps = 0
            self.start_time = None
            self.log_lines.clear()
        # 清空队列
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    # ── 读取（Gradio 主线程调用） ─────────────────────────────────────

    def poll(self) -> List[Dict]:
        """
        非阻塞拉取所有待处理事件，处理并返回。
        Gradio Timer/Generator 每隔 2s 调用一次。
        """
        events = []
        while True:
            try:
                event = self._queue.get_nowait()
                events.append(event)
                self._process_event(event)
            except queue.Empty:
                break
        return events

    def _process_event(self, event: Dict) -> None:
        etype = event.get("type", "metrics")

        if etype == "metrics":
            m = TrainingMetrics(
                step=event.get("step", 0),
                loss=event.get("loss"),
                eval_loss=event.get("eval_loss"),
                learning_rate=event.get("learning_rate"),
                epoch=event.get("epoch"),
                speed=event.get("speed"),
                gpu_mem_gb=event.get("gpu_mem_gb"),
            )
            with self._lock:
                self.history.append(m)
                self.current = m
                self.completed_steps = m.step

        elif etype == "status":
            with self._lock:
                self.status = event.get("status", self.status)
                if event.get("total_steps"):
                    self.total_steps = int(event["total_steps"])
                if self.status == "running" and self.start_time is None:
                    self.start_time = time.time()
                if event.get("error"):
                    self.error_message = event["error"]

        elif etype == "checkpoint":
            path = event.get("path", "")
            with self._lock:
                if path and path not in self.checkpoints:
                    self.checkpoints.append(path)

        elif etype == "log":
            line = event.get("line", "")
            if line:
                with self._lock:
                    self.log_lines.append(line)
                    if len(self.log_lines) > self._max_log_lines:
                        self.log_lines = self.log_lines[-self._max_log_lines:]

    # ── 查询接口（Gradio 主线程调用） ────────────────────────────────

    def get_loss_curve(self) -> Tuple[List[int], List[float]]:
        with self._lock:
            steps = [m.step for m in self.history if m.loss is not None]
            losses = [m.loss for m in self.history if m.loss is not None]
        return steps, losses

    def get_lr_curve(self) -> Tuple[List[int], List[float]]:
        with self._lock:
            steps = [m.step for m in self.history if m.learning_rate is not None]
            lrs = [m.learning_rate for m in self.history if m.learning_rate is not None]
        return steps, lrs

    def get_progress(self) -> Tuple[int, int, float]:
        """返回 (completed_steps, total_steps, percentage)"""
        with self._lock:
            c, t = self.completed_steps, self.total_steps
        if t == 0:
            return c, t, 0.0
        return c, t, min(100.0, c / t * 100)

    def get_eta(self) -> str:
        with self._lock:
            start = self.start_time
            completed = self.completed_steps
            total = self.total_steps
        if not start or completed == 0 or total == 0:
            return "计算中..."
        elapsed = time.time() - start
        speed = completed / elapsed
        remaining = (total - completed) / speed
        if remaining < 60:
            return f"约 {int(remaining)} 秒"
        elif remaining < 3600:
            return f"约 {int(remaining / 60)} 分钟"
        else:
            h = int(remaining / 3600)
            m = int((remaining % 3600) / 60)
            return f"约 {h} 小时 {m} 分钟"

    def get_current_gpu_mem(self) -> str:
        with self._lock:
            cur = self.current
        if cur and cur.gpu_mem_gb:
            return f"{cur.gpu_mem_gb:.1f} GB"
        # 实时查询
        try:
            import torch
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
                return f"{mem:.1f} GB"
        except Exception:
            pass
        return "—"

    def get_current_speed(self) -> str:
        with self._lock:
            cur = self.current
        if cur and cur.speed:
            return f"{cur.speed:.2f} samples/s"
        return "—"

    def get_status_label(self) -> str:
        labels = {
            "idle": "待机",
            "loading": "加载模型中...",
            "running": "训练中",
            "paused": "已暂停",
            "stopped": "已停止",
            "finished": "训练完成",
            "error": f"错误：{self.error_message}",
        }
        return labels.get(self.status, self.status)

    def get_log_text(self) -> str:
        with self._lock:
            return "\n".join(self.log_lines[-100:])

    def get_checkpoints(self) -> List[str]:
        with self._lock:
            return list(self.checkpoints)

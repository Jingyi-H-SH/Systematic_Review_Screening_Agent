import os
import io
import asyncio
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
from agent_system import run_full_screening_process
import json

# 为WebSockets和任务管理设置
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, task_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[task_id] = websocket

    def disconnect(self, task_id: str):
        if task_id in self.active_connections:
            del self.active_connections[task_id]

    async def send_progress(self, task_id: str, message: str):
        if task_id in self.active_connections:
            await self.active_connections[task_id].send_text(message)

manager = ConnectionManager()
tasks = {}

# 加载环境变量
load_dotenv()

# 初始化FastAPI应用
app = FastAPI(
    title="文献筛选智能体Web应用",
    description="一个使用多智能体系统进行文献筛选的交互式Web应用。",
    version="2.0.0"
)

# --- 1. 主页HTML ---
@app.get("/", response_class=HTMLResponse)
async def get_home_page():
    # 主页HTML内容，包含表单、进度显示和结果区域
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>文献筛选智能体</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 0; background-color: #f8f9fa; display: flex; justify-content: center; padding: 2rem; }
            .container { width: 100%; max-width: 900px; background-color: #ffffff; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); padding: 2.5rem; }
            h1 { color: #333; text-align: center; }
            .form-group { margin-bottom: 1.5rem; }
            label { display: block; font-weight: 600; margin-bottom: 0.5rem; color: #555; }
            textarea, input[type="file"] { width: 100%; padding: 0.75rem; border-radius: 8px; border: 1px solid #ced4da; font-size: 1rem; box-sizing: border-box; }
            textarea { height: 200px; resize: vertical; }
            button { width: 100%; padding: 1rem; background-color: #007bff; color: white; border: none; border-radius: 8px; font-size: 1.1rem; font-weight: 600; cursor: pointer; transition: background-color 0.2s; }
            button:hover { background-color: #0056b3; }
            button:disabled { background-color: #a0a0a0; cursor: not-allowed; }
            #status, #results { margin-top: 2rem; padding: 1rem; border-radius: 8px; background-color: #e9ecef; }
            #status h2, #results h2 { margin-top: 0; color: #333; }
            #progress { margin-top: 1rem; }
            #results-table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
            #results-table th, #results-table td { border: 1px solid #dee2e6; padding: 0.75rem; text-align: left; }
            #results-table th { background-color: #f2f2f2; }
            #download-link { display: none; margin-top: 1.5rem; text-align: center; }
            #download-link a { text-decoration: none; background-color: #28a745; color: white; padding: 0.8rem 1.5rem; border-radius: 8px; font-weight: 600; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>文献筛选智能体系统</h1>
            <form id="screening-form">
                <div class="form-group">
                    <label for="criteria">1. 输入筛选标准:</label>
                    <textarea id="criteria" name="criteria" required></textarea>
                </div>
                <div class="form-group">
                    <label for="file">2. 上传待筛选的CSV文件:</label>
                    <input type="file" id="file" name="file" accept=".csv" required>
                </div>
                <button type="submit" id="submit-btn">开始筛选</button>
            </form>
            <div id="status" style="display:none;">
                <h2>处理状态</h2>
                <div id="progress">等待任务开始...</div>
            </div>
            <div id="results" style="display:none;">
                <h2>筛选结果预览 (前5条)</h2>
                <div id="results-table-container"></div>
                <div id="download-link"></div>
            </div>
        </div>
        <script>
            const form = document.getElementById('screening-form');
            const submitBtn = document.getElementById('submit-btn');
            const statusDiv = document.getElementById('status');
            const progressDiv = document.getElementById('progress');
            const resultsDiv = document.getElementById('results');
            const tableContainer = document.getElementById('results-table-container');
            const downloadLinkDiv = document.getElementById('download-link');

            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                submitBtn.disabled = true;
                submitBtn.innerText = '处理中...';
                statusDiv.style.display = 'block';
                resultsDiv.style.display = 'none';
                progressDiv.innerText = '正在上传文件...';
                
                const formData = new FormData(form);
                const task_id = 'task_' + Date.now();

                const ws = new WebSocket(`wss://${window.location.host}/ws/${task_id}`);
                ws.onmessage = (event) => {
                    progressDiv.innerText = event.data;
                };
                ws.onclose = () => {
                    progressDiv.innerText = "连接关闭。";
                };

                try {
                    const response = await fetch(`/screen/?task_id=${task_id}`, {
                        method: 'POST',
                        body: formData,
                    });

                    if (response.ok) {
                        progressDiv.innerText = '✅ 处理完成！正在生成结果...';
                        const result = await response.json();
                        
                        // 显示结果预览
                        const tableHtml = result.preview_html;
                        tableContainer.innerHTML = tableHtml;
                        resultsDiv.style.display = 'block';

                        // 创建下载链接
                        const blob = new Blob([result.full_csv], { type: 'text/csv' });
                        const url = window.URL.createObjectURL(blob);
                        downloadLinkDiv.innerHTML = `<a href="${url}" download="screened_results.csv">下载完整筛选结果</a>`;
                        downloadLinkDiv.style.display = 'block';
                        
                    } else {
                        const error = await response.json();
                        progressDiv.innerText = `❌ 错误: ${error.detail}`;
                    }
                } catch (error) {
                    progressDiv.innerText = `❌ 发生网络错误: ${error.message}`;
                } finally {
                    submitBtn.disabled = false;
                    submitBtn.innerText = '开始筛选';
                    ws.close();
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# --- 2. WebSocket 端点用于进度更新 ---
@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    await manager.connect(task_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(task_id)

# --- 3. 后端处理函数 ---
async def process_screening_task(df, criteria, task_id):
    # 定义进度回调函数
    async def progress_callback(message: str):
        await manager.send_progress(task_id, message)
    
    # 运行核心筛选逻辑
    result_df = await run_full_screening_process(df, criteria, progress_callback)
    
    # 准备结果
    preview_html = result_df.head().to_html(index=False, border=0, classes='table table-striped')
    full_csv = result_df.to_csv(index=False)
    
    tasks[task_id] = {"status": "completed", "result": {"preview_html": preview_html, "full_csv": full_csv}}

# --- 4. 主筛选API端点 ---
@app.post("/screen/")
async def screen_articles_endpoint(task_id: str, criteria: str = Form(...), file: UploadFile = File(...)):
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="文件类型无效，请上传CSV文件。")

    try:
        contents = await file.read()
        buffer = io.StringIO(contents.decode('utf-8'))
        df = pd.read_csv(buffer)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法解析CSV文件: {e}")

    # 在后台运行耗时任务
    asyncio.create_task(process_screening_task(df, criteria, task_id))
    
    # 等待任务完成
    while tasks.get(task_id, {}).get("status") != "completed":
        await asyncio.sleep(1)
        
    result = tasks[task_id]['result']
    del tasks[task_id] # 清理已完成的任务
    return result


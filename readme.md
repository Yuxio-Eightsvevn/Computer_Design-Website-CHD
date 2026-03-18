### 启动命令

```bash
conda activate challengeserver
cd /var/www/Diagnostic_Platform_Stable_Version_C3D/
nohup uvicorn main:app --host 0.0.0.0 --port 11000 --workers 4 > app.log 2>&1 < /dev/null &
```

### 网址

- [登录] http://127.0.0.1:11000/
- [后台] http://127.0.0.1:11000/admin


### 确认运行情况
```bash
ps aux | grep uvicorn | grep -v grep
```

### 关闭命令
```bash
kill + id
pkill -f uvicorn
```
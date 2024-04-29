import subprocess
import gradio as gr

def start():
    subprocess.call("chmod -R 777 ./*",shell=True)
    subprocess.call(f"./frp/frpc -c ./frp/frpc.toml & python ./server.py & python ./monitor/os_monitor.py -p 2334 &",shell=True)
    
with gr.Blocks() as app:
    with gr.Tabs():     
        with gr.TabItem("说明"):
            gr.Markdown("## 注意事项")
            gr.Markdown("1. 用前必看，否则后果自负：[使用规约 & 免责声明](https://www.bilibili.com/read/cv26659988)")
            gr.Markdown("2. 该空间只用于挂载后端，无法进行任何操作")
            gr.Markdown("## 模型信息")
            gr.Markdown("**模型训练：**[红血球AE3803](https://space.bilibili.com/6589795)")
            gr.Markdown("**所用数据集：**[星穹铁道&原神全角色中日英韩语音包/数据集](https://www.bilibili.com/read/cv24180458)")
            gr.Markdown("**所用项目：**[Bert-VITS 2](https://github.com/fishaudio/Bert-VITS2)")
            gr.Markdown("## 相关链接")
            gr.Markdown("**推理后端硬件监测：**[点我传送](https://mon.ai-hobbyist.org/zabbix.php?action=dashboard.view&dashboardid=1&page=3)")
            gr.Markdown("**推理服务状态总览：**[点我传送](https://status.ai-hobbyist.org/status/infer)")
            gr.Markdown("## 相关下载")
            gr.Markdown("**权重下载：**[点我传送](https://pan.ai-hobbyist.org/InferPack/Vits)")
            gr.Markdown("**数据集下载：**[点我传送](https://pan.ai-hobbyist.org/StarRail%20Datasets)")    
app.launch()
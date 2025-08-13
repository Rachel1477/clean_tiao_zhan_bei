# --- START OF FILE gradio_interface.py ---

import gradio as gr
import subprocess
import sys
import threading
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  #当前文件的路径
def create_training_interface():
    """创建并返回Gradio训练界面。"""

    def start_training(*args):
        """
        当用户点击“开始训练”按钮时执行此函数。
        它在一个单独的线程中启动一个子进程来运行训练脚本。
        """
        (data_root_dir, model_save_dir, log_dir, num_epochs, batch_size, 
         learning_rate, patience, max_seq_len, warmup_epochs, grad_clip) = args

        # 禁用按钮，更新状态
        yield {
            status_textbox: gr.update(value="准备中...", interactive=False),
            start_button: gr.update(interactive=False),
            log_textbox: "准备启动训练进程...\n"
        }
        
        # 构建命令行
        command = [
            sys.executable,  # 使用当前Python解释器       
            os.path.join(BASE_DIR,'run_training_process.py'),
            '--data_root_dir', str(data_root_dir),
            '--model_save_dir', str(model_save_dir),
            '--log_dir', str(log_dir),
            '--num_epochs', str(num_epochs),
            '--batch_size', str(batch_size),
            '--learning_rate', str(learning_rate),
            '--patience', str(patience),
            '--max_seq_len', str(max_seq_len),
            '--warmup_epochs', str(warmup_epochs),
            '--grad_clip', str(grad_clip),
        ]
        
        log_output = f"执行命令: {' '.join(command)}\n\n"
        yield {log_textbox: log_output, status_textbox: "训练中..."}

        # 启动子进程
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                bufsize=1  # 行缓冲
            )

            # 实时读取输出
            for line in iter(process.stdout.readline, ''):
                log_output += line
                yield {log_textbox: log_output}
            
            process.stdout.close()
            return_code = process.wait()

            if return_code == 0:
                final_status = "训练成功完成！"
            else:
                final_status = f"训练出错！退出代码: {return_code}"
                
            log_output += f"\n--- 进程已结束 ---\n"
            
            yield {
                log_textbox: log_output,
                status_textbox: gr.update(value=final_status, interactive=False),
                start_button: gr.update(interactive=True) # 重新启用按钮
            }

        except Exception as e:
            log_output += f"\n\n启动训练进程时发生严重错误: {e}"
            yield {
                log_textbox: log_output,
                status_textbox: gr.update(value=f"错误: {e}", interactive=False),
                start_button: gr.update(interactive=True)
            }


    # --- Gradio UI 布局 ---
    with gr.Blocks(theme=gr.themes.Soft(), title="模型训练") as training_iface:
        gr.Markdown("# 模型训练控制台")
        gr.Markdown("在此配置超参数并启动模型训练过程。训练日志将实时显示在下方。")

        with gr.Row():
            status_textbox = gr.Textbox(label="当前状态", value="空闲", interactive=False)
            start_button = gr.Button("开始训练", variant="primary")

        with gr.Accordion("训练配置", open=True):
            with gr.Row():
                data_root_dir = gr.Textbox(label="数据根目录", value=os.path.join(BASE_DIR,'../'))
                model_save_dir = gr.Textbox(label="模型保存目录", value=os.path.join(BASE_DIR,'../src/model_save'))
                log_dir = gr.Textbox(label="日志目录 (TensorBoard)", value=os.path.join(BASE_DIR,'../src/logs'))
            with gr.Row():
                num_epochs = gr.Slider(minimum=1, maximum=200, value=50, step=1, label="训练轮次 (Epochs)")
                batch_size = gr.Slider(minimum=2, maximum=64, value=16, step=2, label="批次大小 (Batch Size)")
                max_seq_len = gr.Slider(minimum=5, maximum=100, value=30, step=1, label="最大序列长度")
            with gr.Row():
                learning_rate = gr.Textbox(label="学习率", value="1e-4")
                patience = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="早停耐心值 (Patience)")
                warmup_epochs = gr.Slider(minimum=0, maximum=20, value=2, step=1, label="预热轮次 (Warmup Epochs)")
                grad_clip = gr.Textbox(label="梯度裁剪阈值", value="1.0")

        gr.Markdown("---")
        log_textbox = gr.Textbox(label=" 训练日志", lines=20, interactive=False, autoscroll=True)

        # 定义按钮点击事件
        inputs = [data_root_dir, model_save_dir, log_dir, num_epochs, batch_size,
                  learning_rate, patience, max_seq_len, warmup_epochs, grad_clip]
        outputs = [status_textbox, start_button, log_textbox]
        
        start_button.click(fn=start_training, inputs=inputs, outputs=outputs)

    return training_iface

def create_prediction_interface():
    """Creates and returns the Gradio prediction interface."""

    def start_prediction(test_data_dir, model_dir, output_dir):
        yield {
            pred_status_textbox: gr.update(value="Preparing...", interactive=False),
            pred_start_button: gr.update(interactive=False),
            pred_log_textbox: "Preparing to start prediction process...\n"
        }

        command = [
            sys.executable,
            os.path.join(BASE_DIR, 'run_prediction_process.py'),
            '--test_data_root', str(test_data_dir),
            '--model_dir', str(model_dir),
            '--output_dir', str(output_dir)
        ]

        log_output = f"Executing command: {' '.join(command)}\n\n"
        yield {pred_log_textbox: log_output, pred_status_textbox: "Prediction in progress..."}

        try:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', bufsize=1
            )
            for line in iter(process.stdout.readline, ''):
                log_output += line
                yield {pred_log_textbox: log_output}
            
            process.stdout.close()
            return_code = process.wait()
            final_status = "Prediction completed successfully!" if return_code == 0 else f"Prediction failed! Exit code: {return_code}"
            
            yield {
                pred_log_textbox: log_output + f"\n--- Process Finished ---\nResults saved to: {output_dir}",
                pred_status_textbox: gr.update(value=final_status),
                pred_start_button: gr.update(interactive=True)
            }
        except Exception as e:
            error_message = f"\n\nFATAL ERROR starting prediction process: {e}"
            yield {
                pred_log_textbox: log_output + error_message,
                pred_status_textbox: gr.update(value=f"Error: {e}"),
                pred_start_button: gr.update(interactive=True)
            }

    with gr.Blocks(theme=gr.themes.Soft()) as prediction_iface:
        gr.Markdown("# Model Prediction Console")
        gr.Markdown("Select the data and model directories, then start the prediction process. Logs will be displayed in real-time.")

        with gr.Row():
            pred_status_textbox = gr.Textbox(label="Current Status", value="Idle", interactive=False)
            pred_start_button = gr.Button("Start Prediction", variant="primary")

        with gr.Accordion("Prediction Configuration", open=True):
            test_data_dir = gr.Textbox(label="Test Data Directory", value=os.path.join(BASE_DIR, '../测试数据'))
            model_dir = gr.Textbox(label="Model Directory (contains best.pth, scaler, etc.)", value=os.path.join(BASE_DIR, '../src/model_save'))
            output_dir = gr.Textbox(label="Output Directory for Results", value=os.path.join(BASE_DIR, '../src/test_results_output'))
            
        pred_log_textbox = gr.Textbox(label="Prediction Log", lines=20, interactive=False, autoscroll=True)

        inputs = [test_data_dir, model_dir, output_dir]
        outputs = [pred_status_textbox, pred_start_button, pred_log_textbox]
        pred_start_button.click(fn=start_prediction, inputs=inputs, outputs=outputs)

    return prediction_iface

# --- Main Application Execution ---
if __name__ == '__main__':
    # Use gr.TabbedInterface to combine the two UIs
    app = gr.TabbedInterface(
        [create_training_interface(), create_prediction_interface()],
        ["Train Model", "Run Prediction"]
    )
  
    print("Launching Gradio App...")
    app.launch()
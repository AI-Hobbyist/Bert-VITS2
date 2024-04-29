import psutil, time, argparse
from flask import Flask, jsonify, request

net_counter = psutil.net_io_counters(pernic=True)
for interface, stats in net_counter.items():
    print(f"网卡：{interface} ,网卡信息：{stats}")

def get_cpu(interval):
    cpu = psutil.cpu_percent(interval)
    return cpu
    
def get_mem():
    mem = psutil.virtual_memory()
    total = mem.total
    available = mem.available
    used = mem.used
    percent = mem.percent
    return total, available, used, percent

def get_swap():
    swap = psutil.swap_memory()
    total = swap.total
    free = swap.free
    used = swap.used
    percent = swap.percent
    return total, free, used, percent

def get_speed(nic):
    s1 = psutil.net_io_counters(pernic=True)[nic]
    time.sleep(1)
    s2 = psutil.net_io_counters(pernic=True)[nic]
    up = s2.bytes_sent - s1.bytes_sent
    dn = s2.bytes_recv - s1.bytes_recv
    sent = s1.bytes_sent
    recv = s1.bytes_recv
    return up, dn, sent, recv

def get_disk(disk):
    disk = psutil.disk_usage(disk)
    total = disk.total
    free = disk.free
    used = disk.used
    percent = disk.percent
    return total, free, used, percent
    
app = Flask(__name__)
@app.route('/hardware', methods=["GET","POST"])
def hardware_status():
    if request.method == "POST":
        data = request.get_json()
        interval = data['interval']
        cpu = get_cpu(interval)
        total_mem, available_mem, used_mem, mem_percent = get_mem()
        total_swap, free_swap, used_swap, swap_percent = get_swap()
        data_dict = {
            "cpu": {
                "usage": cpu
            },
            "mem": {
                "total": total_mem,
                "available": available_mem,
                "used": used_mem,
                "percent": mem_percent
            },
            "swap": {
                "total": total_swap,
                "free": free_swap,
                "used": used_swap,
                "percent": swap_percent
            }
                     }
        return jsonify(data_dict), 200
        
    else:
        dic = {"msg": "不支持的请求方式！"}
        return jsonify(dic), 405
    
@app.route('/network', methods=["GET","POST"])
def network_status():
    if request.method == "POST":
        data = request.get_json()
        nic = data['nic']
        up, dn, sent, recv = get_speed(nic)
        data_dict = {
            "upload": up,
            "download": dn,
            "sent": sent,
            "recv": recv
        }
        return jsonify(data_dict), 200
        
    else:
        dic = {"msg": "不支持的请求方式！"}
        return jsonify(dic), 405
    
@app.route('/disk', methods=["GET","POST"])
def disk_status():
    if request.method == "POST":
        data = request.get_json()
        disk = data['disk']
        total, free, used, percent = get_disk(disk)
        data_dict = {
            "total": total,
            "free": free,
            "used": used,
            "percent": percent
        }
        return jsonify(data_dict), 200
        
    else:
        dic = {"msg": "不支持的请求方式！"}
        return jsonify(dic), 405

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--port',type=int,help='端口号',default=2333)
    args = parser.parse_args()
    app.run(host='0.0.0.0',debug=False,port=args.port)
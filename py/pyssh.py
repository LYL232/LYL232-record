"""
python ssh远程连接执行命令并返回结果
"""
import os
import paramiko

host = '8.8.8.8'
user = 'username'
s = paramiko.SSHClient()
s.load_system_host_keys()
s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
privatekeyfile = os.path.expanduser('/home/username/.ssh/id_rsa')  # 定义key路径
mykey = paramiko.RSAKey.from_private_key_file(privatekeyfile)
s.connect(host, 22, user, pkey=mykey, timeout=5)
stdin, stdout, stderr = s.exec_command('ls ~/')
results = stdout.read().decode('utf8').split('\n')
s.close()
print(results)

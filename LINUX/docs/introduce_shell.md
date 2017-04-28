# SHELL实践

## ssh [secure shell]

```ssh```主要用于：远程加密登录与远程命令执行，替代明文传输的```telnet```命令。

```ssh```采用的是：client-server架构，public-key cryptography机制。

常使用的```ssh```命令形式有：

```
$ ssh login-name@host-name
$ ssh host-name -l login-name
```

相关的配置文件有：

```
/etc/ssh/
/etc/ssh/ssh_config
/etc/ssh/sshd_config
```

查看```sshd```服务是否开启：

```
$ sudo ps -e | grep ssh
```

```/var/log/auth.log```，用于存储```ssh```的登录日志。

参考资料

http://www.openssh.com/

http://man.openbsd.org/ssh.1

https://en.wikipedia.org/wiki/Secure_Shell

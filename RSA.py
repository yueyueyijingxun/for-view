#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:yijing
# datetime:2018/4/1 19:59
# software: PyCharm
#比较麻烦一点的文档

from Crypto import Random
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5 as Cipher_pkcs1_v1_5
import base64
from Crypto.Hash import SHA
from Crypto.Signature import PKCS1_v1_5 as Signature_pkcs1_v1_5
#伪随机数生成器
random_generator=Random.new().read
#rsa算法生成实例
rsa=RSA.generate(1024,random_generator)
#master的秘钥对生成
private_pem=rsa.exportKey()
with open('master-private.pem','wb')as f:
    f.write(private_pem)

public_pem=rsa.publickey().exportKey()
with open('mater-public.pem','wb')as f:
    f.write(public_pem)

#ghost的秘钥对生成
private_pem=rsa.exportKey()
with open('ghost-private.pem','wb')as f:
    f.write(private_pem)

public_pem=rsa.publickey().exportKey()
with open('ghost-public.pem','wb')as f:
    f.write(public_pem)

#用ghost的公钥加密信息
message='WUHAN UNIVERSITY ALL RIGHTS RESERVED'
with open('ghost-public.pem')as f:
    key=f.read()
    rsakey=RSA.importKey(key)
    cipher=Cipher_pkcs1_v1_5.new(rsakey)
    cipher_text=base64.b64encode(cipher.encrypt(message))#完成对message的加密，机密后显示为cipher_text

#储存文件为encodemessage
def save_to_file(file_name, contents):
    fh = open(file_name, 'w')
    fh.write(contents)
    fh.close()
save_to_file('encodemessage.txt', cipher_text)

#ghost使用自己的私钥对内容进行解密
with open('ghost-private.pem')as f:
    key=f.read()
    rsakey=RSA.importKey(key)
    cipher=Cipher_pkcs1_v1_5.new(rsakey)
    text=cipher.decrypt(base64.b64decode(cipher_text), random_generator)
print text

#master使用自己的公钥对内容进行签名
with open('master-private.pem')as f:
    key=f.read()
    rsakey=RSA.importKey(key)
    signer=Signature_pkcs1_v1_5.new(rsakey)
    digest=SHA.new()
    digest.update(message)
    sign=signer.sign(digest)
    signature=base64.b64encode(sign)
save_to_file('signer.txt', signature)
'''
#验证签名
with open('master-public.pem')as f:
    key=f.read()
    rsakey=RSA.importKey(key)
    verifier=Signature_pkcs1_v1_5.new(rsakey)
    digest-SHA.new()
    #假定数据是基于base64编码的
    digest.update(message)
    is_verify=signer.verify(digest,base64.b64decode(signature))
save_to_file('verify.txt',is_verify)
'''
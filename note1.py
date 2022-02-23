import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as ss
import time

#################################################
#生成数据
#生成m*n稀疏矩阵,m较大，n较小,ele表示非0元素个数，表示x
def x_produce(m,n,ele):
    num_row = m
    num_col = n
    num_ele = ele
    a = [np.random.randint(0, num_row) for _ in range(num_ele - 1)] + [num_row - 1]  # random.randint 返回一个随机整型数，范围从低（包括）到高（不包括）,保证出现num_row行
    b = [np.random.randint(0, num_col) for _ in range(num_ele - num_col)] + [i for i in range(num_col)]   # 保证每一列都有值，不会出现全零列
    c = [np.random.rand() for _ in range(num_ele)]  # 返回一个或一组服从“0~1”均匀分布的随机样本值
    rows, cols, v = np.array(a), np.array(b), np.array(c)
    sparseX = ss.coo_matrix((v, (rows, cols)))
    X = sparseX.todense()
    return X

#生成矩阵A(k*m),m较大，ele表示非0元素个数，同时A的每一列为单位向量
def A_produce(k,m,ele):
    num_row = k
    num_col = m
    num_ele = ele
    a = [np.random.randint(0, num_row) for _ in range(num_ele - num_row)] + [i for i in range(num_row)]  # 保证每一行都有值，不会出现全零行
    b = [np.random.randint(0, num_col) for _ in range(num_ele - 1)] + [num_col - 1]  # random.randint 返回一个随机整型数，范围从低（包括）到高（不包括）,保证出现num_col行
    c = [np.random.rand() for _ in range(num_ele)]  # 返回一个或一组服从“0~1”均匀分布的随机样本值
    rows, cols, v = np.array(a), np.array(b), np.array(c)
    sparseX = ss.coo_matrix((v, (rows, cols)))
    X = sparseX.todense()

    #对X的列进行单位化
    X_2 = np.dot(X.T,X)
    for i in range(num_row):
        if X_2[i,i] != 0:
            X[:,i] = X[:,i].copy()/np.sqrt(X_2[i,i])

    return X

# b = np.dot(A,x) 即可得b

#计算矩阵Frobenius 范数
#输入矩阵A
def cal_fro(A):
    A_2 = np.multiply(A,A)
    x = np.sum(A_2)
    return x

#使用q对a正交化
def Schmidt(q,a):
    b = a.copy()
    for i in range(len(q)):
        b = b - np.dot(q[i].T,b)[0,0]*q[i]
    mol = np.sqrt(np.dot(b.T,b)[0,0])
    if mol != 0:
        c = b/np.sqrt(mol)
    else:
        c = b
    return c

########################################################
#阈值
epslion = 1e-30

#m逐渐增大，cpu_time 表示cpu运行时间列表
m_num = []
cpu_time_BMP = []
cpu_time_OMP = []
cpu_time_ORMP = []
#迭代次数
ite_num_BMP = []
ite_num_OMP = []
ite_num_ORMP = []
#计算误差
error_BMP = []
error_OMP = []
error_ORMP = []

for i in np.arange(5,80)*5:
    #维度k,m,n,非0元个数ele1,ele2
    k = 5
    m = 20*i
    n = 3
    ele1 = 5*i
    ele2 = 5*i

    A = A_produce(k,m,ele1)
    x = x_produce(m,n,ele2)
    y = np.dot(A,x)

    #x的范数
    x_fro = cal_fro(x)

    ######################################
    # M-BMP

    #循环索引p，非0行索引I
    p = 0
    I = []  #存储的元素对应索引从0开始
    #计算得到的X,初始值赋为0
    x_result = np.zeros(x.shape)
    #残差
    res = y
    #res范数
    res_norm = cal_fro(res.copy())
    #生成计算残差使用的单位阵
    one = np.eye(k)
    judge = 0
    #计算运行时间
    starttime = time.perf_counter()
    while res_norm >= epslion :
        #储存残差值z_2
        value = []
        for i in range(m):
            z = np.dot(A[:,i].T,res)
            #向量范数
            z_2 = np.dot(z,z.T)
            value.append(z_2[0,0])
            kp = value.index(max(value))   #对应索引从0开始
            #判断kp是否在I中
        if np.any(np.array(value) > 0 ):
            while (kp in I):
                if np.any(np.array(value) > 0):
                    value[kp] = 0
                    kp = value.index(max(value))
                else:
                    judge = 1
                    break

        if judge ==1:
            break
        #将kp存储入I
        I.append(kp)
        res_last = res.copy()
        res = np.dot((one - np.dot(A[:,kp],A[:,kp].T)),res.copy())
        res_norm = cal_fro(res.copy())
        # 计算x中的第kp行
        x_result[kp, :] = np.dot(A[:, kp].T, (res_last-res)).copy()
        p = p+1



    endtime = time.perf_counter()
    cpu = endtime - starttime

    m_num.append(m)
    cpu_time_BMP.append(cpu)
    ite_num_BMP.append(p)
    #计算误差
    error = cal_fro((x - x_result))/x_fro
    error_BMP.append(res_norm)
    np.save('E:\\华为——科研\\note1\\m_num_M_BMP.npy',np.array(m_num))
    np.save('E:\\华为——科研\\note1\\cpu_time_M_BMP.npy', np.array(cpu_time_BMP))
    np.save('E:\\华为——科研\\note1\\ite_num_BMP.npy', np.array(ite_num_BMP))
    np.save('E:\\华为——科研\\note1\\error_BMP.npy', np.array(error_BMP))


    print('m=%d时：'%(m))
    print('\nBMP_CPU运行时间:%.8f s' % (cpu))
    print('\nBMP迭代次数:%d 次' % (p))
    print('\nBMP计算相对误差:%.8f ' % (res_norm))

    ############################################
    #M-OMP
    # 循环索引p，非0行索引I
    p = 0
    I = []  # 存储的元素对应索引从0开始
    # 残差
    res = y
    # res范数
    res_norm = cal_fro(res.copy())
    # 计算得到的X,初始值赋为0
    x_result = np.zeros(x.shape)
    # 取出的a_kp，并正交化后得到的q_p,第一个赋为0
    q = np.zeros((k, 1))
    q_p = []
    q_p.append(q)
    judge = 0
    # 生成计算残差使用的单位阵
    one = np.eye(k)
    # 计算运行时间
    starttime = time.perf_counter()

    while res_norm >= epslion:
        #存储计算max的值
        value = []
        for i in range(m):
            z = np.dot(A[:,i].T,res)
            z_2 = np.dot(z,z.T)
            value.append(z_2[0,0])

        kp = value.index(max(value))  # 对应索引从0开始
        # 判断kp是否在I中
        if np.any(np.array(value) > 0):
            while (kp in I):
                if np.any(np.array(value) > 0):
                    value[kp] = 0
                    kp = value.index(max(value))
                else:
                    judge = 1
                    break

        if judge:
            break
        I.append(kp)
        #使用q_p 对A[:,kp]正交化
        q = Schmidt(q_p,A[:,kp])
        q_p.append(q)
        # 计算x中的第kp行
        x_result[kp, :] = np.dot(A[:, kp].T, np.dot(np.dot(q,q.T),res.copy())).copy()
        res = np.dot((one - np.dot(q,q.T)),res.copy())
        res_norm = cal_fro(res)

        p = p+1

    endtime = time.perf_counter()
    cpu = endtime - starttime

    cpu_time_OMP.append(cpu)
    ite_num_OMP.append(p)
    # 计算误差
    error = cal_fro((x - x_result))/x_fro
    error_OMP.append(res_norm)
    np.save('E:\\华为——科研\\note1\\m_num_M_OMP.npy', np.array(m_num))
    np.save('E:\\华为——科研\\note1\\cpu_time_OMP.npy', np.array(cpu_time_OMP))
    np.save('E:\\华为——科研\\note1\\ite_num_OMP.npy', np.array(ite_num_OMP))
    np.save('E:\\华为——科研\\note1\\error_OMP.npy', np.array(error_OMP))

    print('\nOMP_CPU运行时间:%.8f s' % (cpu))
    print('\nOMP迭代次数:%d 次' % (p))
    print('\nOMP计算相对误差:%.8f ' % (res_norm))

    #############################################
    #M-ORMP

    #循环索引p，非0行索引I
    p = 0
    I = []  #存储的元素对应索引从0开始
    #残差
    res = y
    #res范数
    res_norm = cal_fro(res.copy())
    # 计算得到的X,初始值赋为0
    x_result = np.zeros(x.shape)
    #计算矩阵A每一列范数及其更新值
    A_norm = np.ones((m,1))
    #取出的a_kp，并正交化后得到的q_p,第一个赋为0
    q = np.zeros((k,1))
    q_p = []
    q_p.append(q)
    # 生成计算残差使用的单位阵
    one = np.eye(k)
    judge = 0
    # 计算运行时间
    starttime = time.perf_counter()
    while res_norm >= epslion:
        #存储计算max的值
        value = []
        for i in range(m):
            z = np.dot(A[:,i].T,res)
            z_2 = np.dot(z,z.T)

            A_norm[i] = A_norm[i] - (np.dot(q_p[p].T,A[:,i]))**2

            if A_norm[i] == 0:
                value.append(0)
            else:
                value.append((z_2/A_norm[i])[0,0])

        kp = value.index(max(value))  # 对应索引从0开始
        # 判断kp是否在I中
        if np.any(np.array(value) > 0):
            while (kp in I):
                if np.any(np.array(value) > 0):
                    value[kp] = 0
                    kp = value.index(max(value))
                else:
                    judge = 1
                    break

        if judge:
            break
        I.append(kp)

        #使用q_p 对A[:,kp]正交化
        q = Schmidt(q_p,A[:,kp])
        q_p.append(q)
        res_last = res.copy()
        res = np.dot((one - np.dot(q,q.T)),res.copy())
        # 计算x中的第kp行
        x_result[kp, :] = np.dot(A[:, kp].T, (res_last - res)).copy()
        res_norm = cal_fro(res)

        p = p+1


    endtime = time.perf_counter()
    cpu = endtime - starttime

    cpu_time_ORMP.append(cpu)
    ite_num_ORMP.append(p)
    # 计算误差
    error = cal_fro((x - x_result))/x_fro
    error_ORMP.append(res_norm)
    np.save('E:\\华为——科研\\note1\\m_num_M_ORMP.npy', np.array(m_num))
    np.save('E:\\华为——科研\\note1\\cpu_time_M_ORMP.npy', np.array(cpu_time_ORMP))
    np.save('E:\\华为——科研\\note1\\ite_num_ORMP.npy', np.array(ite_num_ORMP))
    np.save('E:\\华为——科研\\note1\\error_ORMP.npy', np.array(error_ORMP))

    print('\nORMP_CPU运行时间:%.8f s' % (cpu))
    print('\nORMP迭代次数:%d 次' % (p))
    print('\nORMP计算相对误差:%.8f ' % (res_norm))

#绘图
# m_num = np.load('E:\\华为——科研\\note1\\m_num_M_BMP.npy')
# cpu_time_BMP = np.load('E:\\华为——科研\\note1\\cpu_time_M_BMP.npy')
# cpu_time_OMP = np.load('E:\\华为——科研\\note1\\cpu_time_OMP.npy')
# cpu_time_ORMP = np.load('E:\\华为——科研\\note1\\cpu_time_M_ORMP.npy')
plt.figure()
plt.plot(m_num[:43],cpu_time_BMP[:43], label="BMP", linestyle=":")
plt.plot(m_num[:43], cpu_time_OMP[:43],label="OMP", linestyle="--")
plt.plot(m_num[:43], cpu_time_ORMP[:43], label="ORMP", linestyle="-.")
plt.legend()
plt.title("speed of three algorithms")
plt.xlabel("m")
plt.ylabel("seconds")
plt.show()


plt.figure()
plt.plot(m_num,ite_num_BMP, label="BMP", linestyle=":")
plt.plot(m_num, ite_num_OMP,label="OMP", linestyle="--")
plt.plot(m_num, ite_num_ORMP, label="ORMP", linestyle="-.")
plt.legend()
plt.title("number of iteration")
plt.xlabel("m")
plt.ylabel("number")
plt.show()


plt.figure()
plt.plot(m_num,error_BMP, label="BMP", linestyle=":")
plt.plot(m_num, error_OMP,label="OMP", linestyle="--")
plt.plot(m_num, error_ORMP, label="ORMP", linestyle="-.")
plt.legend()
plt.title("error of algorithm")
plt.xlabel("m")
plt.ylabel("error")
plt.show()




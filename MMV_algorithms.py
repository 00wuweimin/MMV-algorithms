import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as ss
import time

#################################################
#sparse matrix
def x_produce(m,n,ele):
    num_row = m
    num_col = n
    num_ele = ele
    a = [np.random.randint(0, num_row) for _ in range(num_ele - 1)] + [num_row - 1]  
    b = [np.random.randint(0, num_col) for _ in range(num_ele - num_col)] + [i for i in range(num_col)]  
    c = [np.random.rand() for _ in range(num_ele)]  
    rows, cols, v = np.array(a), np.array(b), np.array(c)
    sparseX = ss.coo_matrix((v, (rows, cols)))
    X = sparseX.todense()
    return X


def A_produce(k,m,ele):
    num_row = k
    num_col = m
    num_ele = ele
    a = [np.random.randint(0, num_row) for _ in range(num_ele - num_row)] + [i for i in range(num_row)]  
    b = [np.random.randint(0, num_col) for _ in range(num_ele - 1)] + [num_col - 1]  
    c = [np.random.rand() for _ in range(num_ele)]  
    rows, cols, v = np.array(a), np.array(b), np.array(c)
    sparseX = ss.coo_matrix((v, (rows, cols)))
    X = sparseX.todense()


    X_2 = np.dot(X.T,X)
    for i in range(num_row):
        if X_2[i,i] != 0:
            X[:,i] = X[:,i].copy()/np.sqrt(X_2[i,i])

    return X


def cal_fro(A):
    A_2 = np.multiply(A,A)
    x = np.sum(A_2)
    return x


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

epslion = 1e-30


m_num = []
cpu_time_BMP = []
cpu_time_OMP = []
cpu_time_ORMP = []

ite_num_BMP = []
ite_num_OMP = []
ite_num_ORMP = []

error_BMP = []
error_OMP = []
error_ORMP = []

for i in np.arange(5,80)*5:
    k = 5
    m = 20*i
    n = 3
    ele1 = 5*i
    ele2 = 5*i

    A = A_produce(k,m,ele1)
    x = x_produce(m,n,ele2)
    y = np.dot(A,x)


    x_fro = cal_fro(x)

    ######################################
    # M-BMP


    p = 0
    I = []  
    x_result = np.zeros(x.shape)
    res = y
    res_norm = cal_fro(res.copy())
    one = np.eye(k)
    judge = 0
    starttime = time.perf_counter()
    while res_norm >= epslion :
        value = []
        for i in range(m):
            z = np.dot(A[:,i].T,res)
            z_2 = np.dot(z,z.T)
            value.append(z_2[0,0])
            kp = value.index(max(value))   
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
        I.append(kp)
        res_last = res.copy()
        res = np.dot((one - np.dot(A[:,kp],A[:,kp].T)),res.copy())
        res_norm = cal_fro(res.copy())
        x_result[kp, :] = np.dot(A[:, kp].T, (res_last-res)).copy()
        p = p+1



    endtime = time.perf_counter()
    cpu = endtime - starttime

    m_num.append(m)
    cpu_time_BMP.append(cpu)
    ite_num_BMP.append(p)
    
    error = cal_fro((x - x_result))/x_fro
    error_BMP.append(res_norm)
    np.save('E:\\note1\\m_num_M_BMP.npy',np.array(m_num))
    np.save('E:\\note1\\cpu_time_M_BMP.npy', np.array(cpu_time_BMP))
    np.save('E:\\note1\\ite_num_BMP.npy', np.array(ite_num_BMP))
    np.save('E:\\note1\\error_BMP.npy', np.array(error_BMP))


    print('m=%d时：'%(m))
    print('\nBMP_CPU:%.8f s' % (cpu))
    print('\nBMP_iteration:%d 次' % (p))
    print('\nBMP_error:%.8f ' % (res_norm))

    ############################################
    #M-OMP
    p = 0
    I = [] 
    res = y

    res_norm = cal_fro(res.copy())
    x_result = np.zeros(x.shape)
    q = np.zeros((k, 1))
    q_p = []
    q_p.append(q)
    judge = 0
    one = np.eye(k)
    starttime = time.perf_counter()

    while res_norm >= epslion:
        value = []
        for i in range(m):
            z = np.dot(A[:,i].T,res)
            z_2 = np.dot(z,z.T)
            value.append(z_2[0,0])

        kp = value.index(max(value)) 
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
        q = Schmidt(q_p,A[:,kp])
        q_p.append(q)
 
        x_result[kp, :] = np.dot(A[:, kp].T, np.dot(np.dot(q,q.T),res.copy())).copy()
        res = np.dot((one - np.dot(q,q.T)),res.copy())
        res_norm = cal_fro(res)

        p = p+1

    endtime = time.perf_counter()
    cpu = endtime - starttime

    cpu_time_OMP.append(cpu)
    ite_num_OMP.append(p)
    error = cal_fro((x - x_result))/x_fro
    error_OMP.append(res_norm)
    np.save('E:\\note1\\m_num_M_OMP.npy', np.array(m_num))
    np.save('E:\\note1\\cpu_time_OMP.npy', np.array(cpu_time_OMP))
    np.save('E:\\note1\\ite_num_OMP.npy', np.array(ite_num_OMP))
    np.save('E:\\note1\\error_OMP.npy', np.array(error_OMP))

    print('\nOMP_CPU:%.8f s' % (cpu))
    print('\nOMP_iteration:%d 次' % (p))
    print('\nOMP_error:%.8f ' % (res_norm))

    #############################################
    #M-ORMP

    p = 0
    I = []  
    res = y
    res_norm = cal_fro(res.copy())
    x_result = np.zeros(x.shape)
    A_norm = np.ones((m,1))
    q = np.zeros((k,1))
    q_p = []
    q_p.append(q)
    one = np.eye(k)
    judge = 0
    starttime = time.perf_counter()
    while res_norm >= epslion:
        value = []
        for i in range(m):
            z = np.dot(A[:,i].T,res)
            z_2 = np.dot(z,z.T)

            A_norm[i] = A_norm[i] - (np.dot(q_p[p].T,A[:,i]))**2

            if A_norm[i] == 0:
                value.append(0)
            else:
                value.append((z_2/A_norm[i])[0,0])

        kp = value.index(max(value))  
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


        q = Schmidt(q_p,A[:,kp])
        q_p.append(q)
        res_last = res.copy()
        res = np.dot((one - np.dot(q,q.T)),res.copy())
        x_result[kp, :] = np.dot(A[:, kp].T, (res_last - res)).copy()
        res_norm = cal_fro(res)

        p = p+1


    endtime = time.perf_counter()
    cpu = endtime - starttime

    cpu_time_ORMP.append(cpu)
    ite_num_ORMP.append(p)
    error = cal_fro((x - x_result))/x_fro
    error_ORMP.append(res_norm)
    np.save('E:\\note1\\m_num_M_ORMP.npy', np.array(m_num))
    np.save('E:\\note1\\cpu_time_M_ORMP.npy', np.array(cpu_time_ORMP))
    np.save('E:\\note1\\ite_num_ORMP.npy', np.array(ite_num_ORMP))
    np.save('E:\\note1\\error_ORMP.npy', np.array(error_ORMP))

    print('\nORMP_CPU:%.8f s' % (cpu))
    print('\nORMP_iteration:%d 次' % (p))
    print('\nORMP_error:%.8f ' % (res_norm))

#绘图
# m_num = np.load('E:\\note1\\m_num_M_BMP.npy')
# cpu_time_BMP = np.load('E:\\note1\\cpu_time_M_BMP.npy')
# cpu_time_OMP = np.load('E:\\note1\\cpu_time_OMP.npy')
# cpu_time_ORMP = np.load('E:\\note1\\cpu_time_M_ORMP.npy')
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




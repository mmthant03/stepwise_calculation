import numpy as np
import homework1_jpcaltabiano_mmthant as hw1

def test_problem4():
    x = np.array([1, 3, 5])
    y = np.array([2, 4, 6])
    answer = hw1.problem4(x, y)
    print(answer)
    if answer == 44: return True
    else: return False

def test_problem5():
    A = np.array([[1,2,3],[4,5,6]])
    print(hw1.problem5(A))

def test_problem6():
    A = np.array([[1,2,3],[4,5,6]])
    print(hw1.problem6(A))

def test_problem7():
    A = np.array([[1,2,3],[1,2,3],[1,2,3]])
    alpha = 1
    print(hw1.problem7(A, alpha))

def test_problem10():
    A = np.array([[1,2,3],[4,5,6],[7,8,9]])
    c, d, = 3, 6
    answer = hw1.problem10(A, c, d)
    print(answer)

def test_problem11():
    A = np.array([[1,2,3],[1,2,3],[1,2,3]])
    k = 2
    answer = hw1.problem11(A, k)
    print(answer)

def test_problem12():
    A = np.array([[1,2,3],[2,2,2],[3,2,1]])
    x = np.array([1,1,1])
    print(A, x, x.T)
    print(hw1.problem12(A, x))

def test_problem13():
    A = np.array([[1,2,3],[2,2,2],[3,2,1]])
    x = np.array([[1,1,1]])
    print(A, x, x.T)
    print(hw1.problem13(A, x.T))

# test_problem4()
# test_problem5()
# test_problem6()
# test_problem7()
# test_problem10()
test_problem13()
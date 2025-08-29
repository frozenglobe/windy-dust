def bisect_list(a):
    g = lambda a: 2 * 3**-1.5 / H_R[:-1] - k[0] / a
    lb_loc = k[0]/a # lower bound for root finding
    solution_check = np.sum(g(a)>0)
    root_R = []
    if solution_check == 0:
        print('No solution')
        return
    else:
        for j in range(solution_check):
            g_loc = lambda z: z / (1 + (H_R[j]*z)**2) ** (3/2) - k[0]/a
            result = bisect(g_loc, lb_loc, 2**-0.5 / H_R[j], maxiter=2000)
            lb_loc = result-2
            root_R.append(result)
        root_R = np.array(root_R)
        return root_R
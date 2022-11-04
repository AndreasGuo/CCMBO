#! python3
# DNA Contraints
# Continuity, Hairpin, H-Measure, Similarity, Melting Tempurate, GC Content

import numpy as np
from copy import copy

class DNAContraints:
    def __init__(self, CT=2, HT=6, p_min=6, r_min=6, DH=0.17, CH=6, DS=0.17, CS=6):
        self.CT = CT 
        self.HT = HT 
        self.p_min = p_min 
        self.r_min = r_min 
        self.DH = DH 
        self.CH = CH 
        self.DS = DS
        self.CS = CS

    def continuity(self, DNASequences): # 只计算最后一条 减少计算量
        sequence = DNASequences[-1]
        temp_length = 1
        c = 0
        i = 0
        while i<len(sequence)-1:
            if sequence[i] == sequence[i+1]:
                temp_length += 1
            else:
                if temp_length > self.CT:
                    c += pow(temp_length, 2)
                temp_length = 1
            i += 1
        # 处理右侧边界
        if temp_length > self.CT:
            c += pow(temp_length, 2)
        return c 

    # hairpin 辅助函数
    def pinlen(self, p,r,i,l):
        return min(p+i, l-p-i-r)
    
    def bp(self, x, y):
        return 1 if x+y==3 else 0

    def hairpin(self, DNASequences): # 只计算最后一条 减少计算量
        sequence = DNASequences[-1]
        l = len(sequence)
        PINLEN = 3
        value = 0
        for p in range(self.p_min, int((l - self.r_min)/2)+1):
            for r in range(self.r_min, (l-2*p)+1):
                for i in range(l-2*p-r+1):
                    sigma_bp = 0
                    pl = self.pinlen(p,r,i,l)
                    for j in range(pl):
                        sigma_bp += self.bp(x=sequence[p+i-j-1], y=sequence[p+i+r+j])
                    value += sigma_bp if sigma_bp>(pl/PINLEN) else 0
        return value

    # HM and SM ancillary functions
    def shift(self, sequence, i, l):
        shiftedDNA = copy(sequence)
        if i==0: 
            return shiftedDNA
        temp = np.zeros(abs(i))+5
        if i > 0 and i < l:
            shiftedDNA = np.append(temp, shiftedDNA[0:l-i])
            return shiftedDNA
        if i < 0 and i > -l:
            shiftedDNA = np.append(shiftedDNA[abs(i): l], temp)
            return shiftedDNA
        shiftedDNA = np.zeros(l)+5
        return shiftedDNA
    
    def ceq(self, x,y,i,l):
        e = 0
        j = i 
        while j<l and x[j]==y[j]:
            e += 1
            j += 1
        return e

    def s_cont(self, x, y, l):
        sigma_eq, i = 0, 1
        while i<l:
            e = self.ceq(x,y,i,l)
            sigma_eq += e if e>self.CS else 0
            i += e if e>0 else 1
        return sigma_eq
    
    def cbp(self,x, y ,i, l):
        c, j = 0, i
        while j<l and x[j] + y[j] == 3:
            c += 1
            j += 1
        return c

    def h_cont(self, x, y, i, l):
        h = 0
        for i in range(l):
            c = self.cbp(x, y, i, l)
            h += c if c>self.CH else 0
        return h 

    def h_dis(self, x, y, l):
        sigma_bp = 0
        for i in range(l):
            sigma_bp += 1 if x[i] + y[i] == 3 else 0
        return sigma_bp if sigma_bp > self.DH*l/2 else 0
    
    def s_his(self, x, y, l):
        sum = 0
        for i in range(l):
            sum += 1 if x[i]==y[i] else 0
        return sum if sum > self.DS*l else 0

    def hm_and_sm(self, DNASequences):
        if len(DNASequences) == 1:
            calculate_sm = False 
        else:
             calculate_sm = True
        m = len(DNASequences)
        l = len(DNASequences[0])
        GAP1 = np.round(1/4)
        hm = np.zeros(m)
        sm = np.zeros(m)
        for p in range(m):
            H = np.zeros(m)
            S = np.zeros(m)
            sequence = DNASequences[p]
            reversed_sequence = sequence[::-1]
            for j in range(m):
                for g in range(int(GAP1+1)):
                    extended_seq_y = np.append(DNASequences[j], np.append(np.zeros(g)+5, DNASequences[j]))
                    for i in range((-l+1),(l)):
                        shifted_seq_y = self.shift(extended_seq_y, i, 2*l+g)
                        current_hm = self.h_dis(reversed_sequence, shifted_seq_y, l)\
                            + self.h_cont(reversed_sequence, shifted_seq_y, i, l)
                        H[j] = max(H[j], current_hm)
                        if calculate_sm and p!=j:
                            current_sm = self.s_his(sequence, shifted_seq_y, l)\
                                + self.s_cont(sequence, shifted_seq_y, l)
                            S[j] = max(S[j], current_sm)
            hm[p] = sum(H)
            sm[p] = sum(S)
        return hm, sm
        
    


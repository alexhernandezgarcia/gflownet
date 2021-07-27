from multiprocessing import Pool

class someClass(object):
   def __init__(self):
       pass
   def f(self, x):
       return x*x
   def g(self, x):
       return x + 1

   def go(self):
      p = Pool(4)
      sc = p.map(self, range(4))
      printRecord(sc)

   def __call__(self, x):
     return self.f(x)


if __name__ == '__main__':
    sc = someClass()
    sc.go()

for k in range(2,32): # size
  # van der corput
  vdc = file("vdc" + str(k), 'w')
  for i in range(1, k + 1): # row
    for j in range(1, i):
      vdc.write('0')
    vdc.write('1')
    for j in range(i + 1, k + 1):
      vdc.write('0')
    vdc.write('\n')
  vdc.close()

  # larcher pillichshammer
  lp = file("lp" + str(k), 'w')
  for i in range(1, k + 1): # row
    for j in range(1, i):
      lp.write('0')
    for j in range(i, k + 1):
      lp.write('1')
    lp.write('\n')
  lp.close()

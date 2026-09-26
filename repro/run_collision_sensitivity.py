"""Separate domain and tolerance sensitivity of an in-phase residence peak."""
from concurrent.futures import ProcessPoolExecutor
from run_dynamics import prepare_collision, collision, dump
def main():
    prepare_collision(800)
    jobs=[(0.,.0075,200,2e-11,2e-13,5000,'inphase_tight_200',.5),
          (0.,.0075,400,1e-9,1e-11,5000,'inphase_loose_400',.5),
          (0.,.0075,800,2e-11,2e-13,5000,'inphase_tight_800',.5)]
    with ProcessPoolExecutor(max_workers=3) as pool:rows=list(pool.map(collision,jobs))
    dump('collision_inphase_sensitivity.csv',rows)
if __name__=='__main__':main()

## How To Run

git clone으로 코드 복사

```

AL01205606:test-run ykkim$ git clone https://github.com/Naver-AI-Hackathon/nsml.git

'nsml'에 복제합니다...
remote: Enumerating objects: 21, done.
remote: Counting objects: 100% (21/21), done.
remote: Compressing objects: 100% (18/18), done.
remote: Total 21 (delta 4), reused 12 (delta 1), pack-reused 0
오브젝트 묶음 푸는 중: 100% (21/21), 완료.

```
mnint 샘플코드로 nsml환경에서 실행하기

```

AL01205606:test-run ykkim$ cd nsml/mnist-example/
AL01205606:mnist-example ykkim$ ls
README.md	ladder.py	main.py		none_test.py	ops.py		rladder.py	setup.py

AL01205606:mnist-example ykkim$ nsml run -d mnist2
INFO[2019/11/28 17:56:54.218] .nsmlignore check - start                    
INFO[2019/11/28 17:56:54.219] .nsmlignore check - done                     
INFO[2019/11/28 17:56:54.241] file integrity check - start                 
INFO[2019/11/28 17:56:54.242] file integrity check - done                  
INFO[2019/11/28 17:56:54.242] README.md 86 B - start                       
INFO[2019/11/28 17:56:54.242] README.md 86 B - done (1/7 14.29%) (86 B/34 KiB 0.25%) 
INFO[2019/11/28 17:56:54.243] ladder.py 9.8 KiB - start                    
INFO[2019/11/28 17:56:54.243] ladder.py 9.8 KiB - done (2/7 28.57%) (9.9 KiB/34 KiB 29.05%) 
INFO[2019/11/28 17:56:54.243] main.py 8.1 KiB - start                      
INFO[2019/11/28 17:56:54.244] main.py 8.1 KiB - done (3/7 42.86%) (18 KiB/34 KiB 52.99%) 
INFO[2019/11/28 17:56:54.244] none_test.py 4.4 KiB - start                 
INFO[2019/11/28 17:56:54.244] none_test.py 4.4 KiB - done (4/7 57.14%) (22 KiB/34 KiB 65.90%) 
INFO[2019/11/28 17:56:54.244] ops.py 3.6 KiB - start                       
INFO[2019/11/28 17:56:54.244] ops.py 3.6 KiB - done (5/7 71.43%) (26 KiB/34 KiB 76.56%) 
INFO[2019/11/28 17:56:54.244] rladder.py 7.7 KiB - start                   
INFO[2019/11/28 17:56:54.245] rladder.py 7.7 KiB - done (6/7 85.71%) (34 KiB/34 KiB 99.31%) 
INFO[2019/11/28 17:56:54.245] setup.py 239 B - start                       
INFO[2019/11/28 17:56:54.245] setup.py 239 B - done (7/7 100.00%) (34 KiB/34 KiB 100.00%) 
.....
Building docker image. It might take for a while
.......
Session nsmlteam/mnist2/10 is started
```

NSML에서 세션 확인
```
AL01205606:mnist-example ykkim$ nsml ps
Name                        Created         Args    Status    Summary                                                                   Description    # of Models    Size       Type
--------------------------  --------------  ------  --------  ------------------------------------------------------------------------  -------------  -------------  ---------  ------
nsmlteam/mnist2/10          2 minutes ago           Running   epoch=2, epoch_total=10, lr=0.002, test/accuracy=98.09, test/loss=22.868                 3              107.22 MB  normal
```

저장된 모델 확인
```
AL01205606:mnist-example ykkim$ nsml model ls nsmlteam/mnist2/10 
Checkpoint    Last Modified    Elapsed    Summary                                                                                          Size
------------  ---------------  ---------  -----------------------------------------------------------------------------------------------  --------
0             23 minutes ago   27.313     epoch=0, epoch_total=10, test/loss=27.84303783416748, test/accuracy=97.25, step=600, lr=0.002    35.74 MB
1             22 minutes ago   27.531     epoch=1, epoch_total=10, test/loss=24.371407127380373, test/accuracy=97.75, step=1200, lr=0.002  35.74 MB
2             22 minutes ago   28.047     epoch=2, epoch_total=10, test/loss=22.867858848571778, test/accuracy=98.09, step=1800, lr=0.002  35.74 MB
3             21 minutes ago   27.463     epoch=3, epoch_total=10, test/loss=21.682355175018312, test/accuracy=98.25, step=2400, lr=0.002  35.74 MB
4             21 minutes ago   27.708     epoch=4, epoch_total=10, test/loss=20.691930503845214, test/accuracy=98.18, step=3000, lr=0.002  35.74 MB
5             20 minutes ago   27.498     epoch=5, epoch_total=10, test/loss=19.672010765075683, test/accuracy=98.33, step=3600, lr=0.002  35.74 MB
6             20 minutes ago   27.861     epoch=6, epoch_total=10, test/loss=18.552079162597657, test/accuracy=98.41, step=4200, lr=0.002  35.74 MB
7             20 minutes ago   27.347     epoch=7, epoch_total=10, test/loss=17.783723611831665, test/accuracy=98.27, step=4800, lr=0.002  35.74 MB
8             19 minutes ago   27.285     epoch=8, epoch_total=10, test/loss=17.163699474334717, test/accuracy=98.48, step=5400, lr=0.002  35.74 MB
9             19 minutes ago   26.914     epoch=9, epoch_total=10, test/loss=16.7233228969574, test/accuracy=98.55, step=6000, lr=0.002    35.74 MB
```
원하는 모델로 submit 진행 (리드보드에 제출)
```
AL01205606:mnist-example ykkim$ nsml submit nsmlteam/mnist2/10 9
.......
Building docker image. It might take for a while
.........load nsml model takes 0.4842050075531006 seconds
.Infer test set. The inference should be completed within 3600 seconds.
.Infer test set takes 2.9665687084198 seconds
...
Score: 0.9855
Done
AL01205606:mnist-example ykkim$ 
```

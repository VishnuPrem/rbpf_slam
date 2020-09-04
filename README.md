# Rao-Blackwellized Particle Filter for SLAM

This project is a simplified implementation of [this paper](http://people.ee.duke.edu/~lcarin/Lihan9.4.06b.pdf)

In short, RBPF SLAM generates multiples maps of the same environment. Active loop closure is carried out in each SLAM iteration. The following figure shows how some maps that are generated are better than others.

<p align="center"> 
<img src="/fig/rbpf loop closure.png" width = "400">
</p>


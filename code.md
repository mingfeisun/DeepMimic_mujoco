# Convert to python
During Chinese New Year: 
* ~~create character xml for Mujoco~~
* define rewards 
* connect the Mujoco env with the PPO learning method

# Mujoco 
At runtime the positions and orientations of all joints defined in the model are stored in the vector mjData.qpos, in the order in which the appear in the kinematic tree. The linear and angular velocities are stored in the vector mjData.qvel. These two vectors have different dimensionality when free or free or ball joints are used, because such joints represent rotations as unit quaternions.


# Scene update procedure
The inheritance relationship: SceneImitate <--- RLSceneSimChar <--- RLScene & SceneSimChar

In SceneSimChar.cpp, input parameter: time_elapsed
* PreUpdate()
``` C++
void cRLSceneSimChar::PreUpdate(double timestep)
{
    cSceneSimChar::PreUpdate(timestep);

    for (int a = 0; a < GetNumAgents(); ++a)
    {
        const auto& ctrl = mAgentReg.GetAgent(a);
        bool new_action = ctrl->NeedNewAction();
        if (new_action)
        {
            NewActionUpdate(a);
        }
    }
}
void cSceneSimChar::PreUpdate(double timestep)
{
    ClearJointForces();
}
```

* UpdateCharacters()
``` C++
void cSceneImitate::UpdateCharacters(double timestep)
{
    UpdateKinChar(timestep);
    cRLSceneSimChar::UpdateCharacters(timestep);
}
void cSceneSimChar::UpdateCharacters(double time_step)
{
    int num_chars = GetNumChars();
    for (int i = 0; i < num_chars; ++i)
    {
        const auto& curr_char = GetCharacter(i);
        curr_char->Update(time_step);
    }
}
```

``` C++
void cSimCharacter::Update(double timestep)
{
    ClearJointTorques();

    if (HasController())
    {
        mController->Update(timestep);
    }

    // dont clear torques until next frame since they can be useful for visualization
    UpdateJoints();
}
```

DeepMimicCharController.cpp
``` C++
void cDeepMimicCharController::Update(double time_step)
{
    cCharController::Update(time_step);
    UpdateCalcTau(time_step, mTau);
    UpdateApplyTau(mTau);
}
```

Controller inheritance relationship: CtPDController <--- CtController <--- cDeepMimicCharController <--- cCharController
``` C++
void cCtController::UpdateCalcTau(double timestep, Eigen::VectorXd& out_tau)
{
    // use unnormalized phase
    double prev_phase = 0;
    if (mEnablePhaseAction)
    {
        prev_phase = mTime / mCyclePeriod;
        prev_phase += mPhaseOffset;
    }

    cDeepMimicCharController::UpdateCalcTau(timestep, out_tau);

    if (mEnablePhaseAction)
    {
        double phase_rate = GetPhaseRate();
        double tar_phase = prev_phase + timestep * phase_rate;
        mPhaseOffset = tar_phase - mTime / mCyclePeriod;
        mPhaseOffset = std::fmod(mPhaseOffset, 1.0);
    }

    UpdateBuildTau(timestep, out_tau);
}
```

``` C++
void cCtPDController::UpdateBuildTau(double time_step, Eigen::VectorXd& out_tau)
{
	UpdatePDCtrls(time_step, out_tau);
}

void cCtPDController::UpdatePDCtrls(double time_step, Eigen::VectorXd& out_tau)
{
	int num_dof = mChar->GetNumDof();
	out_tau = Eigen::VectorXd::Zero(num_dof);
	mPDCtrl.UpdateControlForce(time_step, out_tau);
}
```

``` C++
void cImpPDController::CalcControlForces(double time_step, Eigen::VectorXd& out_tau)
{
	double t = time_step;

	const Eigen::VectorXd& pose = mChar->GetPose();
	const Eigen::VectorXd& vel = mChar->GetVel();
	Eigen::VectorXd tar_pose;
	Eigen::VectorXd tar_vel;
	BuildTargetPose(tar_pose);
	BuildTargetVel(tar_vel);

	Eigen::DiagonalMatrix<double, Eigen::Dynamic> Kp_mat = mKp.asDiagonal();
	Eigen::DiagonalMatrix<double, Eigen::Dynamic> Kd_mat = mKd.asDiagonal();

	for (int j = 0; j < GetNumJoints(); ++j)
	{
		const cPDController& pd_ctrl = GetPDCtrl(j);
		if (!pd_ctrl.IsValid() || !pd_ctrl.IsActive())
		{
			int param_offset = mChar->GetParamOffset(j);
			int param_size = mChar->GetParamSize(j);
			Kp_mat.diagonal().segment(param_offset, param_size).setZero();
			Kd_mat.diagonal().segment(param_offset, param_size).setZero();
		}
	}

	Eigen::MatrixXd M = mRBDModel->GetMassMat();
	const Eigen::VectorXd& C = mRBDModel->GetBiasForce();
	M.diagonal() += t * mKd;

	Eigen::VectorXd pose_inc;
	const Eigen::MatrixXd& joint_mat = mChar->GetJointMat();
	cKinTree::VelToPoseDiff(joint_mat, pose, vel, pose_inc);

	pose_inc = pose + t * pose_inc;
	cKinTree::PostProcessPose(joint_mat, pose_inc);

	Eigen::VectorXd pose_err;
	cKinTree::CalcVel(joint_mat, pose_inc, tar_pose, 1, pose_err);
	Eigen::VectorXd vel_err = tar_vel - vel;
	Eigen::VectorXd acc = Kp_mat * pose_err + Kd_mat * vel_err - C;
	
#if defined(IMP_PD_CTRL_PROFILER)
	TIMER_RECORD_BEG(Solve)
#endif

	//int root_size = cKinTree::gRootDim;
	//int num_act_dofs = static_cast<int>(acc.size()) - root_size;
	//auto M_act = M.block(root_size, root_size, num_act_dofs, num_act_dofs);
	//auto acc_act = acc.segment(root_size, num_act_dofs);
	//acc_act = M_act.ldlt().solve(acc_act);
	
	acc = M.ldlt().solve(acc);

#if defined(IMP_PD_CTRL_PROFILER)
	TIMER_RECORD_END(Solve, mPerfSolveTime, mPerfSolveCount)
#endif
	
	out_tau += Kp_mat * pose_err + Kd_mat * (vel_err - t * acc);
}
```
``` C++ 
void cDeepMimicCharController::UpdateCalcTau(double timestep, Eigen::VectorXd& out_tau)
{
    mTime += timestep;
    if (mNeedNewAction)
    {
        HandleNewAction();
    }
}
```


SimCharacter.cpp 
``` C++
void cSimCharacter::ApplyControlForces(const Eigen::VectorXd& tau)
{
    assert(tau.size() == GetNumDof());
    for (int j = 1; j < GetNumJoints(); ++j)
    {
        cSimBodyJoint& joint = GetJoint(j);
        if (joint.IsValid())
        {
            int param_offset = GetParamOffset(j);
            int param_size = GetParamSize(j);
            if (param_size > 0)
            {
                Eigen::VectorXd curr_tau = tau.segment(param_offset, param_size);
                joint.AddTau(curr_tau);
            }
        }
    }
}
```

* UpdateWorld()
``` C++
void cSceneSimChar::UpdateWorld(double time_step)
{
    mWorld->Update(time_step);
}
```

* UpdateGround()
``` C++
void cSceneSimChar::UpdateGround(double time_elapsed)
{
    tVector view_min;
    tVector view_max;
    GetViewBound(view_min, view_max);
    mGround->Update(time_elapsed, view_min, view_max);
}
```

* UpdateObjs()
``` C++
void cSceneSimChar::UpdateObjs(double time_step)
{
    int num_objs = GetNumObjs();
    for (int i = 0; i < num_objs; ++i)
    {
        const tObjEntry& obj = mObjs[i];
        if (obj.IsValid() && obj.mEndTime <= GetTime())
        {
            RemoveObj(i);
        }
    }
}
```

* UpdateJoints()
``` C++
void cSceneSimChar::UpdateJoints(double timestep)
{
    int num_joints = GetNumJoints();
    for (int j = 0; j < num_joints; ++j)
    {
        const tJointEntry& joint = mJoints[j];
        if (joint.IsValid())
        {
            joint.mJoint->ApplyTau();
        }
    }
}
```

* PostUpdateCharacters()
``` C++
void cSceneSimChar::PostUpdateCharacters(double time_step)
{
    int num_chars = GetNumChars();
    for (int i = 0; i < num_chars; ++i)
    {
        const auto& curr_char = GetCharacter(i);
        curr_char->PostUpdate(time_step);
    }
}
```

* PostUpdate()
``` C++
void cSceneSimChar::PostUpdate(double timestep)
{
}
```

# State and action definition

* qpos order:
``` xml
<joint name="chest" pos="0 0 0" range="0 68.75" type="ball"/>
<joint name="neck" pos="0 0 0" range="0 57.3" type="ball"/>
<joint name="right_shoulder" pos="0 0 0" range="0 57.3" type="ball"/>
<joint axis="0 1 0" name="right_elbow" pos="0 0 0" range="-150 0" type="hinge"/>
<joint name="left_shoulder" pos="0 0 0" range="0 57.3" type="ball"/>
<joint axis="0 1 0" name="left_elbow" pos="0 0 0" range="-150 0" type="hinge"/>
<joint name="right_hip" pos="0 0 0" range="0 57.3" type="ball"/>
<joint axis="0 1 0" name="right_knee" pos="0 0 0" range="0 150" type="hinge"/>
<joint name="right_ankle" pos="0 0 0" range="0 57.3" type="ball"/>
<joint name="left_hip" pos="0 0 0" range="0 57.3" type="ball"/>
<joint axis="0 1 0" name="left_knee" pos="0 0 0" range="0 150" type="hinge"/>
<joint name="left_ankle" pos="0 0 0" range="0 57.3" type="ball"/>
```

* Action: (angle, axis_x, axis_y, axis_z) --> quaternion for control

**State** 197 = 106 (pos) + 90 (vel) + 1 (phase):
* Size
``` C++
int cCtController::GetStatePoseSize() const
{
    int pos_dim = GetPosFeatureDim();
    int rot_dim = GetRotFeatureDim();
    int size = mChar->GetNumBodyParts() * (pos_dim + rot_dim) + 1; // +1 for root y

    return size;
}

int cCtController::GetStateVelSize() const
{
    int pos_dim = GetPosFeatureDim();
    int rot_dim = GetRotFeatureDim();
    int size = mChar->GetNumBodyParts() * (pos_dim + rot_dim - 1);
    return size;
}
```
* Contents
``` C++
void cCtController::BuildStatePose(Eigen::VectorXd& out_pose) const
{
    tMatrix origin_trans = mChar->BuildOriginTrans();
    tQuaternion origin_quat = cMathUtil::RotMatToQuaternion(origin_trans);

    bool flip_stance = FlipStance();
    if (flip_stance)
    {
        origin_trans.row(2) *= -1; // reflect z
    }

    tVector root_pos = mChar->GetRootPos();
    tVector root_pos_rel = root_pos;

    root_pos_rel[3] = 1;
    root_pos_rel = origin_trans * root_pos_rel;
    root_pos_rel[3] = 0;

    out_pose = Eigen::VectorXd::Zero(GetStatePoseSize());
    out_pose[0] = root_pos_rel[1];
    int num_parts = mChar->GetNumBodyParts();
    int root_id = mChar->GetRootID();

    int pos_dim = GetPosFeatureDim();
    int rot_dim = GetRotFeatureDim();

    tQuaternion mirror_inv_origin_quat = origin_quat.conjugate();
    mirror_inv_origin_quat = cMathUtil::MirrorQuaternion(mirror_inv_origin_quat, cMathUtil::eAxisZ);

    int idx = 1;
    for (int i = 0; i < num_parts; ++i)
    {
        int part_id = RetargetJointID(i);
        if (mChar->IsValidBodyPart(part_id))
        {
            const auto& curr_part = mChar->GetBodyPart(part_id);
            tVector curr_pos = curr_part->GetPos();

            if (mRecordWorldRootPos && i == root_id)
            {
                if (flip_stance)
                {
                    curr_pos = cMathUtil::QuatRotVec(origin_quat, curr_pos);
                    curr_pos[2] = -curr_pos[2];
                    curr_pos = cMathUtil::QuatRotVec(mirror_inv_origin_quat, curr_pos);
                }
            }
            else
            {
                curr_pos[3] = 1;
                curr_pos = origin_trans * curr_pos;
                curr_pos -= root_pos_rel;
                curr_pos[3] = 0;
            }

            out_pose.segment(idx, pos_dim) = curr_pos.segment(0, pos_dim);
            idx += pos_dim;

            tQuaternion curr_quat = curr_part->GetRotation();
            if (mRecordWorldRootRot && i == root_id)
            {
                if (flip_stance)
                {
                    curr_quat = origin_quat * curr_quat;
                    curr_quat = cMathUtil::MirrorQuaternion(curr_quat, cMathUtil::eAxisZ);
                    curr_quat = mirror_inv_origin_quat * curr_quat;
                }
            }
            else
            {
                curr_quat = origin_quat * curr_quat;
                if (flip_stance)
                {
                    curr_quat = cMathUtil::MirrorQuaternion(curr_quat, cMathUtil::eAxisZ);
                }
            }

            if (curr_quat.w() < 0)
            {
                curr_quat.w() *= -1;
                curr_quat.x() *= -1;
                curr_quat.y() *= -1;
                curr_quat.z() *= -1;
            }
            out_pose.segment(idx, rot_dim) = cMathUtil::QuatToVec(curr_quat).segment(0, rot_dim);
            idx += rot_dim;
        }
    }
}

void cCtController::BuildStateVel(Eigen::VectorXd& out_vel) const
{
    int num_parts = mChar->GetNumBodyParts();
    tMatrix origin_trans = mChar->BuildOriginTrans();
    tQuaternion origin_quat = cMathUtil::RotMatToQuaternion(origin_trans);

    bool flip_stance = FlipStance();
    if (flip_stance)
    {
        origin_trans.row(2) *= -1; // reflect z
    }

    int pos_dim = GetPosFeatureDim();
    int rot_dim = GetRotFeatureDim();

    out_vel = Eigen::VectorXd::Zero(GetStateVelSize());

    tQuaternion mirror_inv_origin_quat = origin_quat.conjugate();
    mirror_inv_origin_quat = cMathUtil::MirrorQuaternion(mirror_inv_origin_quat, cMathUtil::eAxisZ);
    
    int idx = 0;
    for (int i = 0; i < num_parts; ++i)
    {
        int part_id = RetargetJointID(i);
        int root_id = mChar->GetRootID();

        const auto& curr_part = mChar->GetBodyPart(part_id);
        tVector curr_vel = curr_part->GetLinearVelocity();

        if (mRecordWorldRootRot && i == root_id)
        {
            if (flip_stance)
            {
                curr_vel = cMathUtil::QuatRotVec(origin_quat, curr_vel);
                curr_vel[2] = -curr_vel[2];
                curr_vel = cMathUtil::QuatRotVec(mirror_inv_origin_quat, curr_vel);
            }
        }
        else
        {
            curr_vel = origin_trans * curr_vel;
        }

        out_vel.segment(idx, pos_dim) = curr_vel.segment(0, pos_dim);
        idx += pos_dim;

        tVector curr_ang_vel = curr_part->GetAngularVelocity();
        if (mRecordWorldRootRot && i == root_id)
        {
            if (flip_stance)
            {
                curr_ang_vel = cMathUtil::QuatRotVec(origin_quat, curr_ang_vel);
                curr_ang_vel[2] = -curr_ang_vel[2];
                curr_ang_vel = -curr_ang_vel;
                curr_ang_vel = cMathUtil::QuatRotVec(mirror_inv_origin_quat, curr_ang_vel);
            }
        }
        else
        {
            curr_ang_vel = origin_trans * curr_ang_vel;
            if (flip_stance)
            {
                curr_ang_vel = -curr_ang_vel;
            }
        }

        out_vel.segment(idx, rot_dim - 1) = curr_ang_vel.segment(0, rot_dim - 1);
        idx += rot_dim - 1;
    }
}
```

**Action** 36 = 4 * 8 (spherical joint) + 4 (revolute joint):
``` C++
int cCtController::GetActionPhaseSize() const
{
    return (mEnablePhaseAction) ? 1 : 0;
}
int cCtController::GetActionCtrlSize() const
{
    int ctrl_size = mChar->GetNumDof();
    int root_size = mChar->GetParamSize(mChar->GetRootID());
    ctrl_size -= root_size;
    return ctrl_size;
}
```

# Structure
``` bash
├── args
│   ├── kin_char_args.txt
│   ├── run_humanoid3d_backflip_args.txt
│   ├── run_humanoid3d_cartwheel_args.txt
│   ├── run_humanoid3d_crawl_args.txt
│   ├── run_humanoid3d_dance_a_args.txt
│   ├── run_humanoid3d_dance_b_args.txt
│   ├── run_humanoid3d_getup_facedown_args.txt
│   ├── run_humanoid3d_getup_faceup_args.txt
│   ├── run_humanoid3d_jump_args.txt
│   ├── run_humanoid3d_kick_args.txt
│   ├── run_humanoid3d_punch_args.txt
│   ├── run_humanoid3d_roll_args.txt
│   ├── run_humanoid3d_run_args.txt
│   ├── run_humanoid3d_spin_args.txt
│   ├── run_humanoid3d_spinkick_args.txt
│   ├── run_humanoid3d_walk_args.txt
│   ├── train_humanoid3d_backflip_args.txt
│   ├── train_humanoid3d_cartwheel_args.txt
│   ├── train_humanoid3d_crawl_args.txt
│   ├── train_humanoid3d_dance_a_args.txt
│   ├── train_humanoid3d_dance_b_args.txt
│   ├── train_humanoid3d_getup_facedown_args.txt
│   ├── train_humanoid3d_getup_faceup_args.txt
│   ├── train_humanoid3d_jump_args.txt
│   ├── train_humanoid3d_kick_args.txt
│   ├── train_humanoid3d_punch_args.txt
│   ├── train_humanoid3d_roll_args.txt
│   ├── train_humanoid3d_run_args.txt
│   ├── train_humanoid3d_spin_args.txt
│   ├── train_humanoid3d_spinkick_args.txt
│   └── train_humanoid3d_walk_args.txt
├── data
│   ├── agents
│   ├── characters
│   ├── controllers
│   ├── motions
│   ├── policies
│   ├── shaders
│   ├── terrain
│   └── textures
├── DeepMimicCore
├── DeepMimic_Optimizer.py
├── DeepMimic.py
├── DeepMimic.pyproj
├── DeepMimic.sln
├── env
│   ├── action_space.py
│   ├── deepmimic_env.py
│   ├── env.py
├── learning
│   ├── agent_builder.py
│   ├── exp_params.py
│   ├── nets
│   ├── normalizer.py
│   ├── path.py
│   ├── pg_agent.py
│   ├── ppo_agent.py
│   ├── replay_buffer.py
│   ├── rl_agent.py
│   ├── rl_util.py
│   ├── rl_world.py
│   ├── solvers
│   ├── tf_agent.py
│   ├── tf_normalizer.py
│   └── tf_util.py
├── libraries
│   ├── bullet3
│   └── eigen
├── mpi_run.py
└── util
    ├── arg_parser.py
    ├── logger.py
    ├── math_util.py
    ├── mpi_util.py
    └── util.py

```

# Code

## Training-related

DeepMimic.py: 
``` python
def build_world(args, enable_draw, playback_speed=1):
    arg_parser = build_arg_parser(args)
    env = DeepMimicEnv(args, enable_draw)
    world = RLWorld(env, arg_parser)
    world.env.set_playback_speed(playback_speed)
    return world
```
``` python
def update_world(world, time_elapsed):
    num_substeps = world.env.get_num_update_substeps()
    timestep = time_elapsed / num_substeps
    num_substeps = 1 if (time_elapsed == 0) else num_substeps

    for i in range(num_substeps):
        world.update(timestep)

        valid_episode = world.env.check_valid_episode()
        if valid_episode:
            end_episode = world.env.is_episode_end()
            if (end_episode):
                world.end_episode()
                world.reset()
                break
        else:
            world.reset()
            break
    return
 ```

rl_world.py
``` python
# world.update()
    def update(self, timestep):
        self._update_agents(timestep)
        self._update_env(timestep)
        return
```

rl_agent.py
``` python
    def update(self, timestep):
        if self.need_new_action():
            self._update_new_action()

        if (self._mode == self.Mode.TRAIN and self.enable_training):
            self._update_counter += timestep

            while self._update_counter >= self.update_period:
                self._train()
                self._update_exp_params()
                self.world.env.set_sample_count(self._total_sample_count)
                self._update_counter -= self.update_period

        return
```
``` python
    def _train(self):
        samples = self.replay_buffer.total_count
        self._total_sample_count = int(MPIUtil.reduce_sum(samples))
        end_training = False
        
        if (self.replay_buffer_initialized):  
            if (self._valid_train_step()):
                prev_iter = self.iter
                iters = self._get_iters_per_update()
                avg_train_return = MPIUtil.reduce_avg(self.train_return)
            
                for i in range(iters):
                    curr_iter = self.iter
                    wall_time = time.time() - self.start_time
                    wall_time /= 60 * 60 # store time in hours

                    has_goal = self.has_goal()
                    s_mean = np.mean(self.s_norm.mean)
                    s_std = np.mean(self.s_norm.std)
                    g_mean = np.mean(self.g_norm.mean) if has_goal else 0
                    g_std = np.mean(self.g_norm.std) if has_goal else 0

                    self.logger.log_tabular("Iteration", self.iter)
                    self.logger.log_tabular("Wall_Time", wall_time)
                    self.logger.log_tabular("Samples", self._total_sample_count)
                    self.logger.log_tabular("Train_Return", avg_train_return)
                    self.logger.log_tabular("Test_Return", self.avg_test_return)
                    self.logger.log_tabular("State_Mean", s_mean)
                    self.logger.log_tabular("State_Std", s_std)
                    self.logger.log_tabular("Goal_Mean", g_mean)
                    self.logger.log_tabular("Goal_Std", g_std)
                    self._log_exp_params()

                    self._update_iter(self.iter + 1)
                    self._train_step()

                    Logger.print("Agent " + str(self.id))
                    self.logger.print_tabular()
                    Logger.print("") 

                    if (self._enable_output() and curr_iter % self.int_output_iters == 0):
                        self.logger.dump_tabular()

                if (prev_iter // self.int_output_iters != self.iter // self.int_output_iters):
                    end_training = self.enable_testing()

        else:

            Logger.print("Agent " + str(self.id))
            Logger.print("Samples: " + str(self._total_sample_count))
            Logger.print("") 

            if (self._total_sample_count >= self.init_samples):
                self.replay_buffer_initialized = True
                end_training = self.enable_testing()
        
        if self._need_normalizer_update:
            self._update_normalizers()
            self._need_normalizer_update = self.normalizer_samples > self._total_sample_count

        if end_training:
            self._init_mode_train_end()
 
        return
```


ppo_agent.py
``` python
    def _train_step(self):
        adv_eps = 1e-5

        start_idx = self.replay_buffer.buffer_tail
        end_idx = self.replay_buffer.buffer_head
        assert(start_idx == 0)
        assert(self.replay_buffer.get_current_size() <= self.replay_buffer.buffer_size) # must avoid overflow
        assert(start_idx < end_idx)

        idx = np.array(list(range(start_idx, end_idx)))        
        end_mask = self.replay_buffer.is_path_end(idx)
        end_mask = np.logical_not(end_mask) 
        
        vals = self._compute_batch_vals(start_idx, end_idx)
        new_vals = self._compute_batch_new_vals(start_idx, end_idx, vals)

        valid_idx = idx[end_mask]
        exp_idx = self.replay_buffer.get_idx_filtered(self.EXP_ACTION_FLAG).copy()
        num_valid_idx = valid_idx.shape[0]
        num_exp_idx = exp_idx.shape[0]
        exp_idx = np.column_stack([exp_idx, np.array(list(range(0, num_exp_idx)), dtype=np.int32)])
        
        local_sample_count = valid_idx.size
        global_sample_count = int(MPIUtil.reduce_sum(local_sample_count))
        mini_batches = int(np.ceil(global_sample_count / self.mini_batch_size))
        
        adv = new_vals[exp_idx[:,0]] - vals[exp_idx[:,0]]
        new_vals = np.clip(new_vals, self.val_min, self.val_max)

        adv_mean = np.mean(adv)
        adv_std = np.std(adv)
        adv = (adv - adv_mean) / (adv_std + adv_eps)
        adv = np.clip(adv, -self.norm_adv_clip, self.norm_adv_clip)

        critic_loss = 0
        actor_loss = 0
        actor_clip_frac = 0

        for e in range(self.epochs):
            np.random.shuffle(valid_idx)
            np.random.shuffle(exp_idx)

            for b in range(mini_batches):
                batch_idx_beg = b * self._local_mini_batch_size
                batch_idx_end = batch_idx_beg + self._local_mini_batch_size

                critic_batch = np.array(range(batch_idx_beg, batch_idx_end), dtype=np.int32)
                actor_batch = critic_batch.copy()
                critic_batch = np.mod(critic_batch, num_valid_idx)
                actor_batch = np.mod(actor_batch, num_exp_idx)
                shuffle_actor = (actor_batch[-1] < actor_batch[0]) or (actor_batch[-1] == num_exp_idx - 1)

                critic_batch = valid_idx[critic_batch]
                actor_batch = exp_idx[actor_batch]
                critic_batch_vals = new_vals[critic_batch]
                actor_batch_adv = adv[actor_batch[:,1]]

                critic_s = self.replay_buffer.get('states', critic_batch)
                critic_g = self.replay_buffer.get('goals', critic_batch) if self.has_goal() else None
                curr_critic_loss = self._update_critic(critic_s, critic_g, critic_batch_vals)

                actor_s = self.replay_buffer.get("states", actor_batch[:,0])
                actor_g = self.replay_buffer.get("goals", actor_batch[:,0]) if self.has_goal() else None
                actor_a = self.replay_buffer.get("actions", actor_batch[:,0])
                actor_logp = self.replay_buffer.get("logps", actor_batch[:,0])
                curr_actor_loss, curr_actor_clip_frac = self._update_actor(actor_s, actor_g, actor_a, actor_logp, actor_batch_adv)
                
                critic_loss += curr_critic_loss
                actor_loss += np.abs(curr_actor_loss)
                actor_clip_frac += curr_actor_clip_frac

                if (shuffle_actor):
                    np.random.shuffle(exp_idx)

        total_batches = mini_batches * self.epochs
        critic_loss /= total_batches
        actor_loss /= total_batches
        actor_clip_frac /= total_batches

        critic_loss = MPIUtil.reduce_avg(critic_loss)
        actor_loss = MPIUtil.reduce_avg(actor_loss)
        actor_clip_frac = MPIUtil.reduce_avg(actor_clip_frac)

        critic_stepsize = self.critic_solver.get_stepsize()
        actor_stepsize = self.update_actor_stepsize(actor_clip_frac)

        self.logger.log_tabular('Critic_Loss', critic_loss)
        self.logger.log_tabular('Critic_Stepsize', critic_stepsize)
        self.logger.log_tabular('Actor_Loss', actor_loss) 
        self.logger.log_tabular('Actor_Stepsize', actor_stepsize)
        self.logger.log_tabular('Clip_Frac', actor_clip_frac)
        self.logger.log_tabular('Adv_Mean', adv_mean)
        self.logger.log_tabular('Adv_Std', adv_std)

        self.replay_buffer.clear()

        return
```

## Reward-related
``` bash
├── anim
│   ├── Character.cpp
│   ├── Character.h
│   ├── KinCharacter.cpp
│   ├── KinCharacter.h
│   ├── KinTree.cpp
│   ├── KinTree.h
│   ├── Motion.cpp
│   ├── Motion.h
│   ├── Shape.cpp
│   └── Shape.h
├── DeepMimicCore.cpp
├── DeepMimicCore.h
├── DeepMimicCore.i
├── DeepMimicCore.py
├── Main.cpp
├── Makefile
├── objs
│   ├── anim
│   ├── DeepMimicCore.o
│   ├── Main.o
│   ├── render
│   ├── scenes
│   ├── sim
│   └── util
├── render
│   ├── Camera.cpp
│   ├── Camera.h
│   ├── DrawCharacter.cpp
│   ├── DrawCharacter.h
│   ├── DrawGround.cpp
│   ├── DrawGround.h
│   ├── DrawKinTree.cpp
│   ├── DrawKinTree.h
│   ├── DrawMesh.cpp
│   ├── DrawMesh.h
│   ├── DrawObj.cpp
│   ├── DrawObj.h
│   ├── DrawPerturb.cpp
│   ├── DrawPerturb.h
│   ├── DrawSimCharacter.cpp
│   ├── DrawSimCharacter.h
│   ├── DrawUtil.cpp
│   ├── DrawUtil.h
│   ├── DrawWorld.cpp
│   ├── DrawWorld.h
│   ├── GraphUtil.cpp
│   ├── GraphUtil.h
│   ├── IBuffer.cpp
│   ├── IBuffer.h
│   ├── lodepng
│   ├── MatrixStack.cpp
│   ├── MatrixStack.h
│   ├── MeshUtil.cpp
│   ├── MeshUtil.h
│   ├── OBJLoader.h
│   ├── RenderState.h
│   ├── Shader.cpp
│   ├── Shader.h
│   ├── ShadowMap.cpp
│   ├── ShadowMap.h
│   ├── TextureDesc.cpp
│   ├── TextureDesc.h
│   ├── TextureUtil.cpp
│   ├── TextureUtil.h
│   ├── VertexBuffer.cpp
│   └── VertexBuffer.h
├── scenes
│   ├── DrawRLScene.cpp
│   ├── DrawRLScene.h
│   ├── DrawScene.cpp
│   ├── DrawScene.h
│   ├── DrawSceneImitate.cpp
│   ├── DrawSceneImitate.h
│   ├── DrawSceneKinChar.cpp
│   ├── DrawSceneKinChar.h
│   ├── DrawSceneSimChar.cpp
│   ├── DrawSceneSimChar.h
│   ├── RLScene.cpp
│   ├── RLScene.h
│   ├── RLSceneSimChar.cpp
│   ├── RLSceneSimChar.h
│   ├── SceneBuilder.cpp
│   ├── SceneBuilder.h
│   ├── Scene.cpp
│   ├── Scene.h
│   ├── SceneImitate.cpp
│   ├── SceneImitate.h
│   ├── SceneKinChar.cpp
│   ├── SceneKinChar.h
│   ├── SceneSimChar.cpp
│   └── SceneSimChar.h
├── sim
│   ├── ActionSpace.h
│   ├── AgentRegistry.cpp
│   ├── AgentRegistry.h
│   ├── CharController.cpp
│   ├── CharController.h
│   ├── ContactManager.cpp
│   ├── ContactManager.h
│   ├── Controller.cpp
│   ├── Controller.h
│   ├── CtController.cpp
│   ├── CtController.h
│   ├── CtCtrlUtil.cpp
│   ├── CtCtrlUtil.h
│   ├── CtPDController.cpp
│   ├── CtPDController.h
│   ├── CtrlBuilder.cpp
│   ├── CtrlBuilder.h
│   ├── CtVelController.cpp
│   ├── CtVelController.h
│   ├── DeepMimicCharController.cpp
│   ├── DeepMimicCharController.h
│   ├── ExpPDController.cpp
│   ├── ExpPDController.h
│   ├── GroundBuilder.cpp
│   ├── GroundBuilder.h
│   ├── Ground.cpp
│   ├── Ground.h
│   ├── GroundPlane.cpp
│   ├── GroundPlane.h
│   ├── ImpPDController.cpp
│   ├── ImpPDController.h
│   ├── MultiBody.cpp
│   ├── MultiBody.h
│   ├── ObjTracer.cpp
│   ├── ObjTracer.h
│   ├── PDController.cpp
│   ├── PDController.h
│   ├── Perturb.cpp
│   ├── Perturb.h
│   ├── PerturbManager.cpp
│   ├── PerturbManager.h
│   ├── RBDModel.cpp
│   ├── RBDModel.h
│   ├── RBDUtil.cpp
│   ├── RBDUtil.h
│   ├── SimBodyJoint.cpp
│   ├── SimBodyJoint.h
│   ├── SimBodyLink.cpp
│   ├── SimBodyLink.h
│   ├── SimBox.cpp
│   ├── SimBox.h
│   ├── SimCapsule.cpp
│   ├── SimCapsule.h
│   ├── SimCharacter.cpp
│   ├── SimCharacter.h
│   ├── SimCharBuilder.cpp
│   ├── SimCharBuilder.h
│   ├── SimCharGeneral.cpp
│   ├── SimCharGeneral.h
│   ├── SimCylinder.cpp
│   ├── SimCylinder.h
│   ├── SimJoint.cpp
│   ├── SimJoint.h
│   ├── SimObj.cpp
│   ├── SimObj.h
│   ├── SimPlane.cpp
│   ├── SimPlane.h
│   ├── SimRigidBody.cpp
│   ├── SimRigidBody.h
│   ├── SimSphere.cpp
│   ├── SimSphere.h
│   ├── SpAlg.cpp
│   ├── SpAlg.h
│   ├── World.cpp
│   └── World.h
└── util
    ├── Annealer.cpp
    ├── Annealer.h
    ├── ArgParser.cpp
    ├── ArgParser.h
    ├── BVHReader.cpp
    ├── BVHReader.h
    ├── CircularBuffer.h
    ├── FileUtil.cpp
    ├── FileUtil.h
    ├── IndexBuffer.h
    ├── IndexManager.cpp
    ├── IndexManager.h
    ├── json
    ├── JsonUtil.cpp
    ├── JsonUtil.h
    ├── MathUtil.cpp
    ├── MathUtil.h
    ├── Rand.cpp
    ├── Rand.h
    ├── Timer.cpp
    ├── Timer.h
    ├── Trajectory.cpp
    └── Trajectory.h

```


SceneImitate.cpp
``` C++
double cSceneImitate::CalcRewardImitate(const cSimCharacter& sim_char, const cKinCharacter& kin_char) const
{
    double pose_w = 0.5;
    double vel_w = 0.05;
    double end_eff_w = 0.15;
    double root_w = 0.2;
    double com_w = 0.1;

    double total_w = pose_w + vel_w + end_eff_w + root_w + com_w;
    pose_w /= total_w;
    vel_w /= total_w;
    end_eff_w /= total_w;
    root_w /= total_w;
    com_w /= total_w;

    const double pose_scale = 2;
    const double vel_scale = 0.1;
    const double end_eff_scale = 40;
    const double root_scale = 5;
    const double com_scale = 10;
    const double err_scale = 1;

    const auto& joint_mat = sim_char.GetJointMat();
    const auto& body_defs = sim_char.GetBodyDefs();
    double reward = 0;

    const Eigen::VectorXd& pose0 = sim_char.GetPose();
    const Eigen::VectorXd& vel0 = sim_char.GetVel();
    const Eigen::VectorXd& pose1 = kin_char.GetPose();
    const Eigen::VectorXd& vel1 = kin_char.GetVel();
    tMatrix origin_trans = sim_char.BuildOriginTrans();
    tMatrix kin_origin_trans = kin_char.BuildOriginTrans();

    tVector com0_world = sim_char.CalcCOM();
    tVector com_vel0_world = sim_char.CalcCOMVel();
    tVector com1_world;
    tVector com_vel1_world;
    cRBDUtil::CalcCoM(joint_mat, body_defs, pose1, vel1, com1_world, com_vel1_world);

    int root_id = sim_char.GetRootID();
    tVector root_pos0 = cKinTree::GetRootPos(joint_mat, pose0);
    tVector root_pos1 = cKinTree::GetRootPos(joint_mat, pose1);
    tQuaternion root_rot0 = cKinTree::GetRootRot(joint_mat, pose0);
    tQuaternion root_rot1 = cKinTree::GetRootRot(joint_mat, pose1);
    tVector root_vel0 = cKinTree::GetRootVel(joint_mat, vel0);
    tVector root_vel1 = cKinTree::GetRootVel(joint_mat, vel1);
    tVector root_ang_vel0 = cKinTree::GetRootAngVel(joint_mat, vel0);
    tVector root_ang_vel1 = cKinTree::GetRootAngVel(joint_mat, vel1);

    double pose_err = 0;
    double vel_err = 0;
    double end_eff_err = 0;
    double root_err = 0;
    double com_err = 0;
    double heading_err = 0;

    int num_end_effs = 0;
    int num_joints = sim_char.GetNumJoints();
    assert(num_joints == mJointWeights.size());

    double root_rot_w = mJointWeights[root_id];
    pose_err += root_rot_w * cKinTree::CalcRootRotErr(joint_mat, pose0, pose1);
    vel_err += root_rot_w * cKinTree::CalcRootAngVelErr(joint_mat, vel0, vel1);

    for (int j = root_id + 1; j < num_joints; ++j)
    {
        double w = mJointWeights[j];
        double curr_pose_err = cKinTree::CalcPoseErr(joint_mat, j, pose0, pose1);
        double curr_vel_err = cKinTree::CalcVelErr(joint_mat, j, vel0, vel1);
        pose_err += w * curr_pose_err;
        vel_err += w * curr_vel_err;

        bool is_end_eff = sim_char.IsEndEffector(j);
        if (is_end_eff)
        {
            tVector pos0 = sim_char.CalcJointPos(j);
            tVector pos1 = cKinTree::CalcJointWorldPos(joint_mat, pose1, j);
            double ground_h0 = mGround->SampleHeight(pos0);
            double ground_h1 = kin_char.GetOriginPos()[1];

            tVector pos_rel0 = pos0 - root_pos0;
            tVector pos_rel1 = pos1 - root_pos1;
            pos_rel0[1] = pos0[1] - ground_h0;
            pos_rel1[1] = pos1[1] - ground_h1;

            pos_rel0 = origin_trans * pos_rel0;
            pos_rel1 = kin_origin_trans * pos_rel1;

            double curr_end_err = (pos_rel1 - pos_rel0).squaredNorm();
            end_eff_err += curr_end_err;
            ++num_end_effs;
        }
    }

    if (num_end_effs > 0)
    {
        end_eff_err /= num_end_effs;
    }

    double root_ground_h0 = mGround->SampleHeight(sim_char.GetRootPos());
    double root_ground_h1 = kin_char.GetOriginPos()[1];
    root_pos0[1] -= root_ground_h0;
    root_pos1[1] -= root_ground_h1;
    double root_pos_err = (root_pos0 - root_pos1).squaredNorm();
    
    double root_rot_err = cMathUtil::QuatDiffTheta(root_rot0, root_rot1);
    root_rot_err *= root_rot_err;

    double root_vel_err = (root_vel1 - root_vel0).squaredNorm();
    double root_ang_vel_err = (root_ang_vel1 - root_ang_vel0).squaredNorm();

    root_err = root_pos_err
            + 0.1 * root_rot_err
            + 0.01 * root_vel_err
            + 0.001 * root_ang_vel_err;
    com_err = 0.1 * (com_vel1_world - com_vel0_world).squaredNorm();

    double pose_reward = exp(-err_scale * pose_scale * pose_err);
    double vel_reward = exp(-err_scale * vel_scale * vel_err);
    double end_eff_reward = exp(-err_scale * end_eff_scale * end_eff_err);
    double root_reward = exp(-err_scale * root_scale * root_err);
    double com_reward = exp(-err_scale * com_scale * com_err);

    reward = pose_w * pose_reward + vel_w * vel_reward + end_eff_w * end_eff_reward
        + root_w * root_reward + com_w * com_reward;

    return reward;
}
```


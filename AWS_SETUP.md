# AWS First-Time Setup for Lasserre SDP d=18 Deploy

One-time steps to get your account ready to launch `x1e.32xlarge` spot
instances for the d=18 Lasserre SDP run.  Total time: **~30 min** if
you've never used AWS before.

---

## 1. Install the AWS CLI (5 min)

**macOS**: `brew install awscli`
**Windows**: Download `https://awscli.amazonaws.com/AWSCLIV2.msi` and run
the installer.  Restart your terminal afterward.
**Linux**: `curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip && unzip awscliv2.zip && sudo ./aws/install`

Verify: `aws --version` should print `aws-cli/2.x.x ...`.

Also install `boto3` in your project venv:
```
pip install boto3
```

---

## 2. Create an IAM user with programmatic access (10 min)

Don't use the root account for day-to-day API access — it's too
privileged.  Create a dedicated user.

1. Log in to the AWS Console → IAM → Users → **Create user**.
2. Name: `sidon-deploy`.
3. Select **Provide user access to the AWS Management Console** = OFF
   (we only want programmatic access, no console login needed).
4. Next → **Attach policies directly** → search for:
   * `AmazonEC2FullAccess`
   (That's enough for launching, attaching EBS, and terminating
    instances.  If you want to be stricter, see the custom policy
    JSON at the end of this file.)
5. Next → Create user.
6. On the user's detail page → **Security credentials** tab → **Access
   keys** → **Create access key** → select **Command Line Interface** →
   Next → **Create access key**.
7. **Copy the Access key ID and Secret access key somewhere safe now**
   — AWS shows the secret exactly once.  If you lose it, make a new
   one; don't stress.

---

## 3. Configure the AWS CLI (2 min)

```
aws configure
```

Paste in:
* **AWS Access Key ID**: (from step 2)
* **AWS Secret Access Key**: (from step 2)
* **Default region name**: `us-east-1` (biggest x1e spot pool; use
  `us-west-2` as fallback if spot capacity is tight there)
* **Default output format**: `json`

Test it:
```
aws sts get-caller-identity
```
Should print your account ID and the IAM user's ARN.  If that works,
you're configured.

---

## 4. Create an EC2 SSH key pair (3 min)

This is the key pair AWS uses to let you SSH into the instance.

```
aws ec2 create-key-pair --key-name sidon-d18 \
    --key-type ed25519 \
    --query KeyMaterial --output text > ~/.ssh/sidon-d18.pem
chmod 600 ~/.ssh/sidon-d18.pem
```

On Windows (Git Bash or WSL), same command.  On native Windows
PowerShell you'd use `(Get-Acl)` permissions — easier to install Git
Bash and do it there.

Verify:
```
ls -la ~/.ssh/sidon-d18.pem
# -rw------- 1 you you 411 ... ~/.ssh/sidon-d18.pem
```

---

## 5. Check x1e.32xlarge availability in your region (1 min)

```
aws ec2 describe-spot-price-history \
    --instance-types x1e.32xlarge \
    --start-time $(date -u -d '1 hour ago' +%FT%TZ 2>/dev/null || date -u -v-1H +%FT%TZ) \
    --product-descriptions "Linux/UNIX" \
    --max-results 5
```

You should see recent prices, typically $3-5/hr.  If the command
returns no entries, x1e is not available in your region — try
`--region us-west-2`.

**Note**: x1e is older hardware (Intel E7v4 Broadwell).  It's
reliable but provisioning can take 2-10 min even when capacity exists.

---

## 6. Subscribe to the base AMI (5 min, one-time)

We use Canonical Ubuntu 22.04 LTS as the base OS.  It's free but needs
an acknowledgement on first use in your account:

```
aws ec2 describe-images \
    --owners 099720109477 \
    --filters 'Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*' \
             'Name=state,Values=available' \
    --query 'sort_by(Images, &CreationDate)[-1].[ImageId,Name]' \
    --output text
```

Note the AMI ID printed (e.g., `ami-0abcdef1234567890`).  You'll pass
this to the deploy script.  **Copy it now**:

```
export SIDON_AMI_ID=ami-0abcdef1234567890
```

(or save it in your `.bashrc` / `.zshrc`)

---

## 7. Prepare your MOSEK license

The deploy script uploads your local `~/mosek/mosek.lic` to the
instance.  Make sure it's there:
```
ls -la ~/mosek/mosek.lic
```
If not, copy from wherever you have it.  AWS does not provide MOSEK
licenses — you must bring your own.

---

## 8. Done — you're ready to launch

```
python deploy_d18_aws_spot.py --ami $SIDON_AMI_ID
```

The deploy script handles everything else: creating a security group,
creating a 10 GB EBS volume for the checkpoint, submitting the spot
request, installing MOSEK and dependencies on the instance, and
running the job with auto-resume on preemption.

---

## Custom IAM policy (optional, tighter than `AmazonEC2FullAccess`)

If you want least-privilege, attach this inline policy to
`sidon-deploy` instead of `AmazonEC2FullAccess`.  It allows only what
the deploy script needs:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ec2:RunInstances",
        "ec2:TerminateInstances",
        "ec2:StopInstances",
        "ec2:StartInstances",
        "ec2:DescribeInstances",
        "ec2:DescribeInstanceStatus",
        "ec2:DescribeImages",
        "ec2:DescribeKeyPairs",
        "ec2:DescribeSubnets",
        "ec2:DescribeVpcs",
        "ec2:DescribeSecurityGroups",
        "ec2:CreateSecurityGroup",
        "ec2:AuthorizeSecurityGroupIngress",
        "ec2:CreateTags",
        "ec2:CreateVolume",
        "ec2:DeleteVolume",
        "ec2:DescribeVolumes",
        "ec2:AttachVolume",
        "ec2:DetachVolume",
        "ec2:DescribeSpotPriceHistory",
        "ec2:RequestSpotInstances",
        "ec2:CancelSpotInstanceRequests",
        "ec2:DescribeSpotInstanceRequests",
        "ec2:CreateFleet",
        "ec2:DeleteFleets",
        "ec2:DescribeFleets"
      ],
      "Resource": "*"
    }
  ]
}
```

---

## Cleanup — how to tear everything down at the end

After the run completes (or you want to abort):

```
python deploy_d18_aws_spot.py --teardown
```

This terminates the instance, deletes the EBS volume, and removes the
security group.  **You will continue to be billed for any resources
you don't clean up** — double-check nothing is running:

```
aws ec2 describe-instances --filters Name=instance-state-name,Values=running,pending --query 'Reservations[].Instances[].[InstanceId,InstanceType]' --output table
```

Empty table = nothing running.  You're good.

---

## Troubleshooting

**"UnauthorizedOperation"**: your IAM user lacks a permission.  Switch
to `AmazonEC2FullAccess` or widen the custom policy.

**"InsufficientInstanceCapacity"**: AWS has no x1e.32xlarge capacity
at that moment.  Retry in 15 min, or switch regions (`--region
us-west-2`).  Spot capacity fluctuates.

**"MaxSpotInstanceCountExceeded"**: your account has no spot quota
for x1e.32xlarge yet.  Go to Console → Service Quotas → EC2 → "All
F/G/P/X/... Spot Instance Requests".  Request an increase to at least
128 vCPUs (x1e.32xlarge has 128 vCPUs).  Approval usually takes <1
business day.  **This is the one step that may delay your first
launch.**  File the quota request BEFORE you're ready to run so
approval has time.

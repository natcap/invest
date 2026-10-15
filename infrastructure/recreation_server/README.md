## Recreation model server infrastructure

Use `playbook.yml` to setup a VM to run an InVEST Recreation Model server process
on the hosts defined in `inventory.ini`.

After creating a debian/ubuntu VM, the playbook will,
* Setup a "staff" group and a "rec" user account
* Install system dependencies
* Clone invest, setup a Python environment with `natcap.invest`
* Mount a bucket with necessary recreation model data using `gcsfuse`
* Stop and start the invest server python process

### Running the ansible playbook
Run the playbook from a control node, a system where `ansible` is installed.
Windows control nodes should use a WSL environment.  
Specify a user that has ssh access from the control node to the VM.
`ansible-playbook -u dfisher5 -i inventory.ini playbook.yml`

By default, the playbook clones the latest `main` from the `natcap` fork of invest.
To install a particular version of invest override these variables by adding
`--extra-vars` to the previous command:

`--extra-vars "invest_version=3.20.2"`
`--extra-vars "invest_version=task/2774-recserver-ansible invest_fork=davemfish"`

If all you want to do is restart the invest server process, it won't hurt to run
the whole playbook, but you may also use this tag to run only the "stop" and "start"
tasks:
`ansible-playbook -u dfisher5 -i inventory.ini playbook.yml --tags restart-process`

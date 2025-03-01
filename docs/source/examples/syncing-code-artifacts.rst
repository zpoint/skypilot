.. _sync-code-artifacts:

Syncing Code and Artifacts
====================================

SkyPilot simplifies transferring code, data, and artifacts to and
from cloud clusters:

- To :ref:`upload code and project files <upload-code-and-project-files>` - use :code:`workdir`

- To :ref:`upload files outside of workdir <file-mounts-example>` (e.g., dotfiles) - use :code:`file_mounts`

- To :ref:`upload/reuse large files <uploading-or-reusing-large-files>` (e.g., datasets) - use :ref:`SkyPilot bucket mounting <sky-storage>`

- To :ref:`download files and artifacts from a cluster <downloading-files-and-artifacts>` - use :ref:`SkyPilot bucket mounting <sky-storage>` or :code:`rsync`

Here, "upload" means uploading files from your local machine (or a cloud object
storage) to a SkyPilot cluster, while "download" means the reverse direction.  The same
mechanisms work for both files and directories.

.. _upload-code-and-project-files:

Uploading code and project files
--------------------------------------
SkyPilot **automatically syncs a local working directory to a cluster** on every
:code:`sky launch` and :code:`sky exec`.  The workdir contains a project's
code and other files, and is typically a Git folder.

The working directory can be configured either

(1) by the :code:`workdir` field in a :ref:`task YAML file <yaml-spec>`, or
(2) by the command line option :code:`--workdir`:

.. code-block:: console

  $ # Assuming task.yaml has a 'workdir: <path>' field, these commands
  $ # sync the workdir to the cluster:
  $ sky launch -c dev task.yaml
  $ sky exec dev task.yaml

  $ # Add a --workdir flag if the yaml doesn't contain the field, or
  $ # to override it.

These commands sync the working directory to :code:`~/sky_workdir` on the remote
VMs.  The task is invoked under that working directory (so that it can call
scripts, access checkpoints, etc.).

.. note::
  To exclude large files from being uploaded, see :ref:`exclude-uploading-files`.

.. note::

  You can keep and edit code in one central place---the local machine where
  :code:`sky` is used---and have them transparently synced to multiple remote
  clusters for execution:

  .. code-block:: console

    $ sky exec cluster0 task.yaml

    $ # Make local edits to the workdir...
    $ # cluster1 will get the updated code.
    $ sky exec cluster1 task.yaml

.. _file-mounts-example:

Uploading files outside of workdir
--------------------------------------

Use the :code:`file_mounts` field in a :ref:`task YAML <yaml-spec>` to upload to a cluster

- local files outside of the working directory (e.g., dotfiles)
- cloud object storage URIs (currently, SkyPilot supports AWS S3, GCP GCS, Cloudflare R2 and IBM COS)

Every :code:`sky launch` invocation reruns the sync up of these files.

Example file mounts:

.. code-block:: yaml

  file_mounts:
    # Format: <cluster path>: <local path/cloud object URI>

    # Upload from local machine to the cluster via rsync.
    /remote/datasets: ~/local/datasets
    ~/.vimrc: ~/.vimrc
    ~/.ssh/id_rsa.pub: ~/.ssh/id_rsa.pub

    # Download from S3 to the cluster.
    /s3-data-test: s3://fah-public-data-covid19-cryptic-pockets/human/il6/PROJ14534/RUN999/CLONE0/results0


For more details, see `this example <https://github.com/skypilot-org/skypilot/blob/master/examples/using_file_mounts.yaml>`_ and :ref:`YAML Configuration <yaml-spec>`.

If you have edited the ``file_mounts`` section and would like to have it reflected on an existing cluster without rerunning the ``setup`` commands,
pass the ``--no-setup`` flag to ``sky launch``. For example, ``sky launch --no-setup -c <cluster_name> <task.yaml>``.

.. note::

    Items listed in a :code:`.skyignore` file under the local file_mount source 
    are also ignored (the same behavior as handling ``workdir``).

.. note::

    If relative paths are used in :code:`file_mounts` or :code:`workdir`, they
    are evaluated relative to the location from which the :code:`sky` command
    is run.

.. _uploading-or-reusing-large-files:

Uploading or reusing large files
--------------------------------------

For large files (e.g., 10s or 100s of GBs), putting them into the workdir or a
file_mount may be slow because they are processed by ``rsync``.  Use
:ref:`SkyPilot bucket mounting <sky-storage>` to efficiently handle
large files.

.. _exclude-uploading-files:

Exclude uploading files
--------------------------------------
By default, SkyPilot uses your existing :code:`.gitignore` and :code:`.git/info/exclude` to exclude files from syncing.

Alternatively, you can use :code:`.skyignore` if you want to separate SkyPilot's syncing behavior from Git's.
If you use a :code:`.skyignore` file, SkyPilot will only exclude files based on that file without using the default Git files.

Any :code:`.skyignore` file under either your workdir or source paths of file_mounts is respected.

:code:`.skyignore` follows RSYNC filter rules, e.g.

.. code-block::

  # Files that match pattern under CURRENT directory
  /file.txt
  /dir
  /*.jar
  /dir/*.jar

  # Files that match pattern under ALL directories
  *.jar
  file.txt

Do _not_ use ``.`` to indicate local directory (e.g., instead of ``./file``, write ``/file``).

.. _downloading-files-and-artifacts:

Downloading files and artifacts
--------------------------------------

Task artifacts, such as **logs and checkpoints**, can either be
transparently uploaded to a cloud object storage, or directly copied from the
remote cluster.

Writing artifacts to cloud object storage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to handling datasets and other task inputs,
:ref:`SkyPilot bucket mounting <sky-storage>` can also be used to directly upload artifacts
generated by tasks. This is achieved by creating a :code:`MOUNT` mode Storage
mount like so in your task YAML:

.. code-block:: yaml

    file_mounts:
      /outputs:
        name: my-sky-outputs    # Can be existing S3 bucket or a new bucket
        store: s3
        mode: MOUNT

This :code:`file_mount` will mount the bucket :code:`s3://my-sky-outputs/`
(creating it if it doesn't exist) at :code:`/outputs`. Since this is specified
with :code:`mode: MOUNT`, any files written to :code:`/outputs` will also be
automatically written to the :code:`s3://my-sky-outputs/` bucket.

Thus, if you point your code to produce files at :code:`/outputs/`, they
will be available on the S3 bucket when they are written to :code:`/outputs/`.
You can then fetch those files either using the `S3 web console <https://s3.console.aws.amazon.com/s3/buckets>`_ or aws-cli
(e.g., :code:`aws s3 ls my-sky-outputs`).


Transferring directly with rsync
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Files and artifacts can also be directly transferred from remote clusters to the
local machine.

To transfer files from cluster nodes, use :code:`rsync` (or :code:`scp`):

.. code-block:: console

  $ # Rsync from head
  $ rsync -Pavz dev:/path/to/checkpoints local/

  $ # Rsync from worker nodes (1-based indexing)
  $ rsync -Pavz dev-worker1:/path/to/checkpoints local/

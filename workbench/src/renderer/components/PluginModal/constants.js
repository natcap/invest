// id for elements related to manual install form
export const manualInstallID = "manualInstall";

// values for sourceType, stored in settingsStore and
// used to determine whether to log a Registry install
export const sourceTypeLocal = "local_path";
export const sourceTypeURL = "git_url";
export const sourceTypeRegistry = "registry";

// addRemoveState opType options:
export const opTypeInstall = "install";
export const opTypeUninstall = "uninstall";

// addRemoveState opStatus options:
export const opStatusLoading = "loading";
export const opStatusSuccess = "success";
export const opStatusFailure = "failure";

// constants used by Plugin Registry components when determining
// whether a plugin is already installed or not
export const thisVersionInstalled = "thisVersionInstalled";
export const anotherVersionInstalled = "anotherVersionInstalled";
export const notInstalled = "notInstalled";

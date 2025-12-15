# OpenAI guided generation requires one of the following:
# - a JSON schema
# - a pydantic class
# For more information, see:
# https://platform.openai.com/docs/guides/structured-outputs#supported-schemas


_TYPES = [
    "string",
    "number",
    "boolean",
    "integer",
    "object",
    "array",
    "enum",
]

_COMBINATIONS = [
    "anyOf",
]

_STRING_PROPERTIES = [
    "pattern",
    "format",
]
_STRING_FORMATS = [
    "date-time",
    "time",
    "date",
    "duration",
    "email",
    "hostname",
    "ipv4",
    "ipv6",
    "uuid",
]

_NUMBER_PROPERTIES = [
    "multipleOf",
    "maximum",
    "exclusiveMaximum",
    "minimum",
    "exclusiveMinimum",
]

_ARRAY_PROPERTIES = [
    "minItems",
    "maxItems",
]

_UNSUPPORTED_TYPE_KEYWORDS = [
    "allOf",
    "not",
    "dependentRequired",
    "dependentSchemas",
    "if",
    "then",
    "else",
]

_UNSUPPORTED_FINE_TUNING_KEYWORDS = [
    "minLength",
    "maxLength",
    "pattern",
    "format",
    "minimum",
    "maximum",
    "multipleOf",
    "patternProperties",
    "minItems",
    "maxItems",
]


def _check_type(type_arg: str | list[str]):
    if isinstance(type_arg, str):
        if type_arg not in _TYPES:
            raise OpenAIInvalidSchema(f"Invalid schema: {type_arg} is not a supported type")
    elif isinstance(type_arg, list):
        nulls = 0
        for t in type_arg:
            if t == "null":
                nulls += 1
            elif t not in _TYPES:
                raise OpenAIInvalidSchema(f"Invalid schema: {t} is not a supported type")
        if nulls == len(type_arg):
            raise OpenAIInvalidSchema("Invalid schema: all types are null")
    else:
        raise OpenAIInvalidSchema(f"Invalid schema: {type_arg} is not a supported type")


def _check_properties(properties: dict[str, dict], type: str | list[str]):
    for k, v in properties.items():
        if type == "string" or (isinstance(type, list) and "string" in type):
            if k not in _STRING_PROPERTIES:
                raise OpenAIInvalidSchema(f"Invalid schema: {k} is not a supported string property")
            if k == "format" and v not in _STRING_FORMATS:
                raise OpenAIInvalidSchema(f"Invalid schema: {v} is not a supported string format")
        if type == "number" or (isinstance(type, list) and "number" in type):
            if k not in _NUMBER_PROPERTIES:
                raise OpenAIInvalidSchema(f"Invalid schema: {k} is not a supported number property")
        if type == "array" or (isinstance(type, list) and "array" in type):
            if k not in _ARRAY_PROPERTIES:
                raise OpenAIInvalidSchema(f"Invalid schema: {k} is not a supported array property")


def _check_required(required: list[str], properties: dict[str, dict]):
    # All fields must be required
    # https://platform.openai.com/docs/guides/structured-outputs#all-fields-must-be-required
    if not all(pk in required for pk, _ in properties.items()):
        raise OpenAIInvalidSchema(
            "Invalid schema: all properties must be required, use an additional null in the type field for optional properties"
        )


def _check_additional_properties(additional_properties: bool):
    # additionalProperties: false must always be set in objects
    # https://platform.openai.com/docs/guides/structured-outputs#additionalproperties-false-must-always-be-set-in-objects
    if additional_properties:
        raise OpenAIInvalidSchema("Invalid schema: additionalProperties must be false")


def _check_string(n_s: int):
    # The total size of all strings in the schema must be less than 15,000 characters
    # https://platform.openai.com/docs/guides/structured-outputs#limitations-on-total-string-size
    if n_s >= 15000:
        raise OpenAIInvalidSchema(
            "Invalid schema: the total size of all strings in the schema must be less than 15,000 characters"
        )


def _check_enum(n_e: int):
    # Limitations on enum size
    # https://platform.openai.com/docs/guides/structured-outputs#limitations-on-enum-size
    if n_e > 500:
        raise OpenAIInvalidSchema(
            "Invalid schema: a schema may have up to 500 enum values across all enum properties"
        )


def _check_string_enum(n_e: int, n_s: int):
    # Limitations on enum size
    # https://platform.openai.com/docs/guides/structured-outputs#limitations-on-enum-size
    if n_e > 250 and n_s > 7500:
        raise OpenAIInvalidSchema(
            "Invalid schema: for a single enum property with string values, the total string length of all enum values cannot exceed 7,500 characters when there are more than 250 enum values"
        )


def _check_num_object_properties(n_o: int):
    # A schema may have up to 100 object properties total
    # https://platform.openai.com/docs/guides/structured-outputs#objects-have-limitations-on-nesting-depth-and-size
    if n_o > 100:
        raise OpenAIInvalidSchema("Invalid schema: a schema may have up to 100 objects")


def _check_object_property_nesting_depth(n_d: int):
    # Limitations on nesting depth
    # https://platform.openai.com/docs/guides/structured-outputs#objects-have-limitations-on-nesting-depth-and-size
    if n_d > 5:
        raise OpenAIInvalidSchema("Invalid schema: a schema may have up to 5 levels of nesting")


def check_schema(schema, root=True, fine_tuned=False) -> tuple[int, int, int, int] | None:
    total_string_length = 0
    total_num_enum_values = 0
    total_num_object_properties = 0
    total_object_property_nesting_depth = 0

    if isinstance(schema, dict):
        for k, v in schema.items():
            if k == "anyOf" and root:
                # Root objects must not be anyOf and must be an object
                # https://platform.openai.com/docs/guides/structured-outputs#root-objects-must-not-be-anyof-and-must-be-an-object
                raise OpenAIInvalidSchema(
                    "Invalid schema: the anyOf object is not supported at the root level"
                )

            if k in _UNSUPPORTED_TYPE_KEYWORDS:
                # Some type-specific keywords are not yet supported
                # https://platform.openai.com/docs/guides/structured-outputs#some-type-specific-keywords-are-not-yet-supported
                raise OpenAIInvalidSchema(f"Invalid schema: {k} is not supported")

            if k in _UNSUPPORTED_FINE_TUNING_KEYWORDS and fine_tuned:
                # Some type-specific keywords are not yet supported for fine-tuned models
                # https://platform.openai.com/docs/guides/structured-outputs#some-type-specific-keywords-are-not-yet-supported
                raise OpenAIInvalidSchema(
                    f"Invalid schema: {k} is not supported for fine-tuned models"
                )

            if k == "type":
                _check_type(v)

            if k == "additionalProperties":
                _check_additional_properties(v)

            if k == "required":
                _check_required(v, schema["properties"])

            if k == "properties":
                _check_properties(v, schema["type"])

                for pk in v.keys():
                    total_string_length += len(str(pk))

                    if schema["type"] == "object":
                        total_num_object_properties += 1
                        total_object_property_nesting_depth += 1

            if k in ("definitions", "$defs") and isinstance(v, dict):
                for dk in v.keys():
                    total_string_length += len(str(dk))

            if k == "enum" and isinstance(v, list):
                enum_value_count = len(v)
                enum_string_length = 0

                for ev in v:
                    if isinstance(ev, str):
                        n = len(ev)
                        enum_string_length += n

                _check_string_enum(enum_value_count, enum_string_length)

                total_num_enum_values += enum_value_count
                total_string_length += enum_string_length

            if k == "const" and isinstance(v, str):
                total_string_length += len(v)

            n_s, n_e, n_o, n_d = check_schema(v, root=False, fine_tuned=fine_tuned)  # type: ignore
            total_string_length += n_s
            total_num_enum_values += n_e
            total_num_object_properties += n_o
            total_object_property_nesting_depth += n_d

    elif isinstance(schema, list):
        if root:
            raise OpenAIInvalidSchema("Invalid schema: arrays are not supported at the root level")
        for item in schema:
            n_s, n_e, n_o, n_d = check_schema(item, root=False, fine_tuned=fine_tuned)  # type: ignore
            total_string_length += n_s
            total_num_enum_values += n_e
            total_num_object_properties += n_o
            total_object_property_nesting_depth += n_d

    if not root:
        return (
            total_string_length,
            total_num_enum_values,
            total_num_object_properties,
            total_object_property_nesting_depth,
        )
    else:
        _check_string(total_string_length)
        _check_enum(total_num_enum_values)
        _check_num_object_properties(total_num_object_properties)
        _check_object_property_nesting_depth(total_object_property_nesting_depth)


class OpenAIInvalidSchema(ValueError):
    pass
